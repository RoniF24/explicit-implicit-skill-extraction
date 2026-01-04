"""
Demo script: compare a single text across all 3 models.
Outputs results to a TXT file with analysis.

Usage:
  python compare_models_demo.py                    # Use default demo text
  python compare_models_demo.py --text "Your custom job description here"
  python compare_models_demo.py --file input.txt  # Read text from file
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/models -> src -> SkillSight

# Default demo text
DEFAULT_DEMO_TEXT = """
I am a senior software engineer with 5 years of experience in Python and Django.
I have built scalable REST APIs and deployed microservices using Docker and Kubernetes.
Strong knowledge of SQL databases including PostgreSQL.
Experience with AWS cloud services and CI/CD pipelines.
"""

# Ground truth for the default demo (what we expect)
DEFAULT_GROUND_TRUTH = {
    "Python": 1.0,       # EXPLICIT
    "Django": 1.0,       # EXPLICIT
    "REST API Design": 1.0,  # EXPLICIT
    "Docker": 1.0,       # EXPLICIT
    "Kubernetes": 1.0,   # EXPLICIT
    "PostgreSQL": 1.0,   # EXPLICIT
    "AWS": 1.0,          # EXPLICIT
    "SQL": 1.0,          # EXPLICIT
    "Microservices": 0.5,  # IMPLICIT (mentioned but not as a skill directly)
}


def load_skills():
    path = REPO_ROOT / "src" / "skills" / "skills_v1.txt"
    return [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def get_all_skills_set():
    return set(load_skills())


# ============================================
# MODEL 1 & 2: Pairwise (RoBERTa / DeBERTa)
# ============================================

def predict_pairwise(model_dir: Path, text: str, skills: list):
    """
    Pairwise model: for each skill, run (skill, text) through the model.
    Returns dict of {skill: score} where score in {0, 0.5, 1.0}
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import warnings
    import logging
    
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    id_to_score = {0: 0.0, 1: 0.5, 2: 1.0}
    results = {}
    
    with torch.no_grad():
        for skill in skills:
            enc = tokenizer(
                skill, text,
                truncation=True,
                max_length=384,
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            out = model(**enc)
            pred_id = torch.argmax(out.logits, dim=-1).item()
            score = id_to_score[pred_id]
            results[skill] = score
    
    return results


# ============================================
# MODEL 3: MODELV2 One-Pass
# ============================================

def predict_modelv2(exp_dir: Path, base_name: str, text: str):
    """
    One-pass model: single forward pass returns all skill predictions.
    Returns dict of {skill: score}
    """
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
    import warnings
    warnings.filterwarnings("ignore")
    
    # Load skills used by MODELV2
    skills_path = REPO_ROOT / "MODELV2" / "splits_v1" / "skills_used.txt"
    if not skills_path.exists():
        return None, "skills_used.txt not found"
    
    skills_used = [s.strip() for s in skills_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    num_skills = len(skills_used)
    
    best_dir = exp_dir / "model_best"
    if not best_dir.exists():
        return None, "model_best not found"
    
    tokenizer = AutoTokenizer.from_pretrained(str(best_dir), use_fast=True)
    
    # Recreate model architecture
    class OnePassSkillClassifier(nn.Module):
        def __init__(self, base_name: str, num_skills: int, dropout: float = 0.1):
            super().__init__()
            self.base = AutoModel.from_pretrained(base_name)
            hidden = self.base.config.hidden_size
            self.num_skills = num_skills
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, num_skills * 3)
        
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            x = self.dropout(cls)
            logits = self.head(x)
            logits = logits.view(-1, self.num_skills, 3)
            return {"logits": logits}
    
    model = OnePassSkillClassifier(base_name, num_skills)
    
    # Load weights
    safetensors_path = best_dir / "model.safetensors"
    bin_path = best_dir / "pytorch_model.bin"
    state = None
    
    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file as safe_load
            state = safe_load(str(safetensors_path))
        except:
            pass
    
    if state is None and bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
    
    if state is None:
        return None, "No weights found"
    
    model.load_state_dict(state, strict=False)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    enc = tokenizer(
        text,
        truncation=True,
        max_length=384,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out["logits"][0]  # [num_skills, 3]
        preds = torch.argmax(logits, dim=-1).tolist()
    
    id_to_score = {0: 0.0, 1: 0.5, 2: 1.0}
    results = {}
    
    for skill, pred_id in zip(skills_used, preds):
        results[skill] = id_to_score[pred_id]
    
    return results, None


def calculate_metrics(predictions: dict, ground_truth: dict):
    """
    Calculate precision, recall, F1 for detected skills.
    """
    # What model detected (score > 0)
    detected = {k for k, v in predictions.items() if v > 0}
    # What should have been detected
    expected = {k for k, v in ground_truth.items() if v > 0}
    
    true_positives = detected & expected
    false_positives = detected - expected
    false_negatives = expected - detected
    
    precision = len(true_positives) / len(detected) if detected else 0
    recall = len(true_positives) / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Score accuracy (for correctly detected skills, was the score right?)
    score_matches = 0
    for skill in true_positives:
        if predictions.get(skill, 0) == ground_truth.get(skill, 0):
            score_matches += 1
    score_accuracy = score_matches / len(true_positives) if true_positives else 0
    
    return {
        "detected": sorted(detected),
        "expected": sorted(expected),
        "true_positives": sorted(true_positives),
        "false_positives": sorted(false_positives),
        "false_negatives": sorted(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "score_accuracy": score_accuracy,
    }


def format_predictions(predictions: dict, top_k: int = 20):
    """Format predictions for display."""
    non_zero = [(k, v) for k, v in predictions.items() if v > 0]
    non_zero.sort(key=lambda x: (-x[1], x[0]))  # Sort by score desc, then name
    return non_zero[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Compare skill extraction across models")
    parser.add_argument("--text", type=str, default=None, help="Custom text to analyze")
    parser.add_argument("--file", type=str, default=None, help="Read text from file")
    parser.add_argument("--output", type=str, default="comparison_results.txt", help="Output file")
    parser.add_argument("--ground-truth", type=str, default=None, 
                        help="JSON file with ground truth {skill: score}")
    args = parser.parse_args()
    
    # Get text to analyze
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
        ground_truth = None
    elif args.text:
        text = args.text
        ground_truth = None
    else:
        text = DEFAULT_DEMO_TEXT
        ground_truth = DEFAULT_GROUND_TRUTH
    
    # Load custom ground truth if provided
    if args.ground_truth:
        ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    
    skills = load_skills()
    all_skills_set = get_all_skills_set()
    
    # Validate ground truth skills are in our list
    if ground_truth:
        invalid_skills = set(ground_truth.keys()) - all_skills_set
        if invalid_skills:
            print(f"WARNING: These skills in ground truth are NOT in skills_v1.txt: {invalid_skills}")
    
    output_lines = []
    
    def log(line=""):
        print(line)
        output_lines.append(line)
    
    log("=" * 70)
    log("SKILL EXTRACTION MODEL COMPARISON")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)
    log()
    log("INPUT TEXT:")
    log("-" * 70)
    log(text.strip())
    log("-" * 70)
    log()
    
    if ground_truth:
        log("GROUND TRUTH (expected skills):")
        log("-" * 70)
        for skill, score in sorted(ground_truth.items(), key=lambda x: (-x[1], x[0])):
            label = "EXPLICIT" if score == 1.0 else "IMPLICIT"
            log(f"  {skill}: {score} ({label})")
        log("-" * 70)
        log()
    
    # Check which models exist
    trained_models_dir = Path(__file__).resolve().parent / "trained_models"
    roberta_dir = trained_models_dir / "roberta_base"
    deberta_dir = trained_models_dir / "deberta_v3_base"
    modelv2_dir = REPO_ROOT / "MODELV2" / "experiments" / "microsoft__deberta-v3-base__onepass"
    
    results = {}
    
    # ---- RoBERTa Pairwise ----
    log("=" * 70)
    log("MODEL 1: RoBERTa Pairwise")
    log("Path: src/models/trained_models/roberta_base")
    log("=" * 70)
    if roberta_dir.exists() and (roberta_dir / "model.safetensors").exists():
        preds = predict_pairwise(roberta_dir, text, skills)
        results["roberta"] = preds
        formatted = format_predictions(preds)
        if formatted:
            for skill, score in formatted:
                label = "EXPLICIT" if score == 1.0 else "IMPLICIT"
                log(f"  {skill}: {score} ({label})")
        else:
            log("  (No skills detected)")
        
        if ground_truth:
            metrics = calculate_metrics(preds, ground_truth)
            log()
            log("  METRICS:")
            log(f"    Precision: {metrics['precision']:.2%}")
            log(f"    Recall:    {metrics['recall']:.2%}")
            log(f"    F1 Score:  {metrics['f1']:.2%}")
            log(f"    Score Accuracy: {metrics['score_accuracy']:.2%}")
            log(f"    True Positives:  {metrics['true_positives']}")
            log(f"    False Positives: {metrics['false_positives']}")
            log(f"    False Negatives: {metrics['false_negatives']}")
    else:
        log("  [SKIPPED - model not found]")
    log()
    
    # ---- DeBERTa Pairwise ----
    log("=" * 70)
    log("MODEL 2: DeBERTa Pairwise")
    log("Path: src/models/trained_models/deberta_v3_base")
    log("=" * 70)
    if deberta_dir.exists() and (deberta_dir / "model.safetensors").exists():
        preds = predict_pairwise(deberta_dir, text, skills)
        results["deberta"] = preds
        formatted = format_predictions(preds)
        if formatted:
            for skill, score in formatted:
                label = "EXPLICIT" if score == 1.0 else "IMPLICIT"
                log(f"  {skill}: {score} ({label})")
        else:
            log("  (No skills detected)")
        
        if ground_truth:
            metrics = calculate_metrics(preds, ground_truth)
            log()
            log("  METRICS:")
            log(f"    Precision: {metrics['precision']:.2%}")
            log(f"    Recall:    {metrics['recall']:.2%}")
            log(f"    F1 Score:  {metrics['f1']:.2%}")
            log(f"    Score Accuracy: {metrics['score_accuracy']:.2%}")
            log(f"    True Positives:  {metrics['true_positives']}")
            log(f"    False Positives: {metrics['false_positives']}")
            log(f"    False Negatives: {metrics['false_negatives']}")
    else:
        log("  [SKIPPED - model not found]")
    log()
    
    # ---- MODELV2 One-Pass ----
    log("=" * 70)
    log("MODEL 3: MODELV2 One-Pass (DeBERTa)")
    log("Path: MODELV2/experiments/microsoft__deberta-v3-base__onepass")
    log("=" * 70)
    if modelv2_dir.exists():
        preds, error = predict_modelv2(modelv2_dir, "microsoft/deberta-v3-base", text)
        if error:
            log(f"  ERROR: {error}")
        elif preds:
            results["modelv2"] = preds
            formatted = format_predictions(preds)
            if formatted:
                for skill, score in formatted:
                    label = "EXPLICIT" if score == 1.0 else "IMPLICIT"
                    log(f"  {skill}: {score} ({label})")
            else:
                log("  (No skills detected) <-- LOW RECALL PROBLEM")
            
            if ground_truth:
                metrics = calculate_metrics(preds, ground_truth)
                log()
                log("  METRICS:")
                log(f"    Precision: {metrics['precision']:.2%}")
                log(f"    Recall:    {metrics['recall']:.2%}")
                log(f"    F1 Score:  {metrics['f1']:.2%}")
                log(f"    Score Accuracy: {metrics['score_accuracy']:.2%}")
                log(f"    True Positives:  {metrics['true_positives']}")
                log(f"    False Positives: {metrics['false_positives']}")
                log(f"    False Negatives: {metrics['false_negatives']}")
    else:
        log("  [SKIPPED - experiment not found]")
    log()
    
    # ---- Summary ----
    log("=" * 70)
    log("SUMMARY COMPARISON")
    log("=" * 70)
    
    if ground_truth and results:
        log()
        log("Model               | Precision | Recall | F1 Score | Detected | Missed")
        log("-" * 70)
        
        for name, preds in results.items():
            metrics = calculate_metrics(preds, ground_truth)
            detected = len([v for v in preds.values() if v > 0])
            missed = len(metrics["false_negatives"])
            log(f"{name:18} | {metrics['precision']:9.1%} | {metrics['recall']:6.1%} | {metrics['f1']:8.1%} | {detected:8} | {missed}")
        
        log()
    
    log("GLOBAL SKILLS LIST INFO:")
    log(f"  Total skills in skills_v1.txt: {len(skills)}")
    log()
    
    log("NOTE: All detected skills MUST be from the global skills list.")
    log("If a skill is not in skills_v1.txt, it cannot be detected.")
    log()
    
    # Save to file
    output_path = REPO_ROOT / args.output
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n>>> Results saved to: {output_path}")


if __name__ == "__main__":
    main()
