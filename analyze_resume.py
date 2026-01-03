"""
Skill extraction demo for resume chunks.
Outputs detailed analysis with percentages.

Usage:
  python analyze_resume.py --text "Your resume text here"
  python analyze_resume.py --text "..." --model deberta
  python analyze_resume.py --text "..." --model all
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import torch

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results_of_model"
RESULTS_DIR.mkdir(exist_ok=True)


def load_skills():
    path = REPO_ROOT / "src" / "skills" / "skills_v1.txt"
    return [s.strip() for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


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
            enc = tokenizer(skill, text, truncation=True, max_length=384, padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            pred_id = torch.argmax(out.logits, dim=-1).item()
            results[skill] = id_to_score[pred_id]
    
    return results


def calculate_metrics(predictions: dict, ground_truth: dict):
    detected = {k for k, v in predictions.items() if v > 0}
    expected = {k for k, v in ground_truth.items() if v > 0}
    
    tp = detected & expected
    fp = detected - expected
    fn = expected - detected
    
    # Detailed breakdown
    explicit_expected = {k for k, v in ground_truth.items() if v == 1.0}
    implicit_expected = {k for k, v in ground_truth.items() if v == 0.5}
    
    explicit_detected = {k for k, v in predictions.items() if v == 1.0}
    implicit_detected = {k for k, v in predictions.items() if v == 0.5}
    
    # Score match (did we get the right score for correctly detected skills?)
    score_correct = 0
    score_wrong = 0
    for skill in tp:
        if predictions[skill] == ground_truth[skill]:
            score_correct += 1
        else:
            score_wrong += 1
    
    precision = len(tp) / len(detected) if detected else 0
    recall = len(tp) / len(expected) if expected else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": sorted(tp),
        "false_positives": sorted(fp),
        "false_negatives": sorted(fn),
        "explicit_expected": sorted(explicit_expected),
        "implicit_expected": sorted(implicit_expected),
        "explicit_detected": sorted(explicit_detected),
        "implicit_detected": sorted(implicit_detected),
        "score_correct": score_correct,
        "score_wrong": score_wrong,
        "total_expected": len(expected),
        "total_detected": len(detected),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze resume text for skills")
    parser.add_argument("--text", type=str, required=True, help="Resume text to analyze")
    parser.add_argument("--model", type=str, default="deberta", choices=["roberta", "deberta", "all"])
    parser.add_argument("--ground-truth", type=str, default=None, help="JSON string or file path with expected skills")
    args = parser.parse_args()
    
    skills = load_skills()
    text = args.text
    
    # Parse ground truth if provided (can be JSON string or file path)
    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        else:
            ground_truth = json.loads(args.ground_truth)
    
    # Determine which models to run
    models_to_run = []
    roberta_dir = REPO_ROOT / "models" / "roberta_base"
    deberta_dir = REPO_ROOT / "models" / "deberta_v3_base"
    
    if args.model == "all":
        if roberta_dir.exists():
            models_to_run.append(("RoBERTa", roberta_dir))
        if deberta_dir.exists():
            models_to_run.append(("DeBERTa", deberta_dir))
        model_suffix = "all_models"
    elif args.model == "roberta":
        models_to_run.append(("RoBERTa", roberta_dir))
        model_suffix = "roberta"
    else:
        models_to_run.append(("DeBERTa", deberta_dir))
        model_suffix = "deberta"
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"analysis_{model_suffix}_{timestamp}.txt"
    
    output_lines = []
    
    def log(line=""):
        print(line)
        output_lines.append(line)
    
    log("=" * 80)
    log("RESUME SKILL EXTRACTION ANALYSIS")
    log("=" * 80)
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Model(s): {', '.join([m[0] for m in models_to_run])}")
    log(f"Output: {output_file.name}")
    log("=" * 80)
    log()
    
    log("RESUME TEXT:")
    log("-" * 80)
    log(text)
    log("-" * 80)
    log()
    
    if ground_truth:
        explicit_gt = [(k, v) for k, v in ground_truth.items() if v == 1.0]
        implicit_gt = [(k, v) for k, v in ground_truth.items() if v == 0.5]
        
        log("GROUND TRUTH (expected skills):")
        log("-" * 80)
        log(f"  EXPLICIT (1.0): {', '.join([k for k, v in explicit_gt])}")
        log(f"  IMPLICIT (0.5): {', '.join([k for k, v in implicit_gt])}")
        log(f"  Total: {len(ground_truth)} skills")
        log("-" * 80)
        log()
    
    for model_name, model_dir in models_to_run:
        log("=" * 80)
        log(f"MODEL: {model_name}")
        log(f"Path: {model_dir.relative_to(REPO_ROOT)}")
        log("=" * 80)
        log()
        
        if not model_dir.exists() or not (model_dir / "model.safetensors").exists():
            log("  [ERROR] Model not found!")
            log()
            continue
        
        preds = predict_pairwise(model_dir, text, skills)
        
        # Separate by type
        explicit = [(k, v) for k, v in preds.items() if v == 1.0]
        implicit = [(k, v) for k, v in preds.items() if v == 0.5]
        
        explicit.sort(key=lambda x: x[0])
        implicit.sort(key=lambda x: x[0])
        
        log("DETECTED SKILLS:")
        log()
        
        if explicit:
            log(f"  EXPLICIT (1.0) - {len(explicit)} skills:")
            for skill, _ in explicit:
                log(f"    • {skill}")
        else:
            log("  EXPLICIT (1.0): None detected")
        
        log()
        
        if implicit:
            log(f"  IMPLICIT (0.5) - {len(implicit)} skills:")
            for skill, _ in implicit:
                log(f"    • {skill}")
        else:
            log("  IMPLICIT (0.5): None detected")
        
        log()
        log(f"  TOTAL DETECTED: {len(explicit) + len(implicit)} skills")
        log()
        
        # Calculate percentages
        total_skills = len(skills)
        detected_count = len(explicit) + len(implicit)
        explicit_pct = len(explicit) / detected_count * 100 if detected_count > 0 else 0
        implicit_pct = len(implicit) / detected_count * 100 if detected_count > 0 else 0
        
        log("STATISTICS:")
        log("-" * 40)
        log(f"  Total skills in vocabulary: {total_skills}")
        log(f"  Skills detected: {detected_count} ({detected_count/total_skills*100:.1f}% of vocabulary)")
        log(f"  Explicit: {len(explicit)} ({explicit_pct:.1f}% of detected)")
        log(f"  Implicit: {len(implicit)} ({implicit_pct:.1f}% of detected)")
        log()
        
        if ground_truth:
            metrics = calculate_metrics(preds, ground_truth)
            
            log("ACCURACY ANALYSIS vs GROUND TRUTH:")
            log("-" * 40)
            log(f"  Precision: {metrics['precision']:.1%} (of detected, how many were correct)")
            log(f"  Recall:    {metrics['recall']:.1%} (of expected, how many were found)")
            log(f"  F1 Score:  {metrics['f1']:.1%} (harmonic mean)")
            log()
            log(f"  True Positives ({len(metrics['true_positives'])}): {', '.join(metrics['true_positives'])}")
            log(f"  False Positives ({len(metrics['false_positives'])}): {', '.join(metrics['false_positives'])}")
            log(f"  False Negatives ({len(metrics['false_negatives'])}): {', '.join(metrics['false_negatives'])}")
            log()
            log(f"  Score Accuracy: {metrics['score_correct']}/{metrics['score_correct']+metrics['score_wrong']} correct scores for true positives")
            log()
    
    log("=" * 80)
    log("END OF ANALYSIS")
    log("=" * 80)
    
    # Save to file
    output_file.write_text("\n".join(output_lines), encoding="utf-8")
    print(f"\n>>> Results saved to: {output_file}")


if __name__ == "__main__":
    main()
