"""
Batch analysis: analyze multiple texts from a JSONL file.
Each line should be: {"text": "...", "ground_truth": {...}}  (ground_truth optional)

Usage:
  python batch_analyze.py --input examples.jsonl --output batch_results.txt
  python batch_analyze.py --input data/splits_v1/test.jsonl --limit 5
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import torch

REPO_ROOT = Path(__file__).resolve().parent


def load_skills():
    path = REPO_ROOT / "src" / "skills" / "skills_v1.txt"
    return [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


# ============================================
# MODEL 1 & 2: Pairwise (RoBERTa / DeBERTa)
# ============================================

def predict_pairwise(model_dir: Path, text: str, skills: list):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import warnings
    warnings.filterwarnings("ignore")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
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
            results[skill] = id_to_score[pred_id]
    
    return results


# ============================================
# MODEL 3: MODELV2 One-Pass
# ============================================

def predict_modelv2(exp_dir: Path, base_name: str, text: str):
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
    import warnings
    warnings.filterwarnings("ignore")
    
    skills_path = REPO_ROOT / "MODELV2" / "splits_v1" / "skills_used.txt"
    if not skills_path.exists():
        return None
    
    skills_used = [s.strip() for s in skills_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    num_skills = len(skills_used)
    
    best_dir = exp_dir / "model_best"
    if not best_dir.exists():
        return None
    
    tokenizer = AutoTokenizer.from_pretrained(str(best_dir), use_fast=True)
    
    class OnePassSkillClassifier(nn.Module):
        def __init__(self, base_name: str, num_skills: int):
            super().__init__()
            self.base = AutoModel.from_pretrained(base_name)
            hidden = self.base.config.hidden_size
            self.num_skills = num_skills
            self.dropout = nn.Dropout(0.1)
            self.head = nn.Linear(hidden, num_skills * 3)
        
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            x = self.dropout(cls)
            logits = self.head(x)
            logits = logits.view(-1, self.num_skills, 3)
            return {"logits": logits}
    
    model = OnePassSkillClassifier(base_name, num_skills)
    
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
        return None
    
    model.load_state_dict(state, strict=False)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    enc = tokenizer(text, truncation=True, max_length=384, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        preds = torch.argmax(out["logits"][0], dim=-1).tolist()
    
    id_to_score = {0: 0.0, 1: 0.5, 2: 1.0}
    return {skill: id_to_score[p] for skill, p in zip(skills_used, preds)}


def calculate_metrics(predictions: dict, ground_truth: dict):
    detected = {k for k, v in predictions.items() if v > 0}
    expected = {k for k, v in ground_truth.items() if v > 0}
    
    tp = detected & expected
    fp = detected - expected
    fn = expected - detected
    
    precision = len(tp) / len(detected) if detected else 0
    recall = len(tp) / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": len(tp), "fp": len(fp), "fn": len(fn)}


def format_predictions(predictions: dict, top_k: int = 10):
    non_zero = [(k, v) for k, v in predictions.items() if v > 0]
    non_zero.sort(key=lambda x: (-x[1], x[0]))
    return non_zero[:top_k]


def load_models():
    """Load all models once."""
    skills = load_skills()
    models = {}
    
    roberta_dir = REPO_ROOT / "models" / "roberta_base"
    deberta_dir = REPO_ROOT / "models" / "deberta_v3_base"
    modelv2_dir = REPO_ROOT / "MODELV2" / "experiments" / "microsoft__deberta-v3-base__onepass"
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import warnings
    warnings.filterwarnings("ignore")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if roberta_dir.exists() and (roberta_dir / "model.safetensors").exists():
        tok = AutoTokenizer.from_pretrained(str(roberta_dir))
        mod = AutoModelForSequenceClassification.from_pretrained(str(roberta_dir))
        mod.eval().to(device)
        models["roberta"] = {"tokenizer": tok, "model": mod, "device": device}
    
    if deberta_dir.exists() and (deberta_dir / "model.safetensors").exists():
        tok = AutoTokenizer.from_pretrained(str(deberta_dir))
        mod = AutoModelForSequenceClassification.from_pretrained(str(deberta_dir))
        mod.eval().to(device)
        models["deberta"] = {"tokenizer": tok, "model": mod, "device": device}
    
    return models, skills


def predict_with_loaded_model(model_info, text: str, skills: list):
    """Predict using pre-loaded model (faster for batch)."""
    tokenizer = model_info["tokenizer"]
    model = model_info["model"]
    device = model_info["device"]
    
    id_to_score = {0: 0.0, 1: 0.5, 2: 1.0}
    results = {}
    
    with torch.no_grad():
        for skill in skills:
            enc = tokenizer(skill, text, truncation=True, max_length=384, padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            pred_id = torch.argmax(out.logits, dim=-1).item()
            results[skill] = id_to_score[pred_id]
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch analyze multiple texts")
    parser.add_argument("--input", type=str, required=True, help="JSONL file with texts")
    parser.add_argument("--output", type=str, default="batch_results.txt", help="Output file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--model", type=str, default="all", choices=["roberta", "deberta", "modelv2", "all"])
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        input_path = REPO_ROOT / args.input
    
    if not input_path.exists():
        print(f"ERROR: File not found: {args.input}")
        return
    
    # Load examples
    examples = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support both formats
            text = obj.get("text") or obj.get("job_description", "")
            gt = obj.get("ground_truth") or obj.get("skills", {})
            if text:
                examples.append({"text": text, "ground_truth": gt})
    
    if args.limit:
        examples = examples[:args.limit]
    
    print(f"Loaded {len(examples)} examples from {input_path}")
    print("Loading models...")
    
    models, skills = load_models()
    print(f"Loaded models: {list(models.keys())}")
    
    output_lines = []
    
    def log(line=""):
        print(line)
        output_lines.append(line)
    
    log("=" * 80)
    log("BATCH SKILL EXTRACTION ANALYSIS")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Input: {input_path}")
    log(f"Examples: {len(examples)}")
    log("=" * 80)
    log()
    
    # Aggregate metrics
    agg_metrics = {name: {"tp": 0, "fp": 0, "fn": 0} for name in models}
    
    for i, ex in enumerate(examples):
        text = ex["text"]
        gt = ex["ground_truth"]
        
        log(f"\n{'='*80}")
        log(f"EXAMPLE {i+1}/{len(examples)}")
        log(f"{'='*80}")
        log(f"TEXT (first 200 chars): {text[:200]}...")
        log()
        
        if gt:
            gt_skills = [(k, v) for k, v in gt.items() if v > 0]
            gt_skills.sort(key=lambda x: (-x[1], x[0]))
            log(f"GROUND TRUTH: {', '.join([f'{s}({v})' for s, v in gt_skills[:10]])}")
            log()
        
        for name, model_info in models.items():
            if args.model != "all" and args.model != name:
                continue
            
            preds = predict_with_loaded_model(model_info, text, skills)
            formatted = format_predictions(preds)
            
            log(f"  {name.upper()}: {', '.join([f'{s}({v})' for s, v in formatted[:8]])}")
            
            if gt:
                metrics = calculate_metrics(preds, gt)
                agg_metrics[name]["tp"] += metrics["tp"]
                agg_metrics[name]["fp"] += metrics["fp"]
                agg_metrics[name]["fn"] += metrics["fn"]
                log(f"    -> P={metrics['precision']:.1%} R={metrics['recall']:.1%} F1={metrics['f1']:.1%}")
    
    # Summary
    log()
    log("=" * 80)
    log("AGGREGATE METRICS (across all examples)")
    log("=" * 80)
    log()
    log(f"{'Model':<15} | {'Precision':>10} | {'Recall':>10} | {'F1 Score':>10} | {'TP':>6} | {'FP':>6} | {'FN':>6}")
    log("-" * 80)
    
    for name, agg in agg_metrics.items():
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        log(f"{name:<15} | {precision:>10.1%} | {recall:>10.1%} | {f1:>10.1%} | {tp:>6} | {fp:>6} | {fn:>6}")
    
    log()
    
    # Save
    output_path = REPO_ROOT / args.output
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n>>> Results saved to: {output_path}")


if __name__ == "__main__":
    main()
