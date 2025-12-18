# eval/run_model_eval.py
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR
from src.datasets.synthetic_jsonl import load_jsonl_dataset
from eval.metrics import (
    binarize_any_evidence,
    binarize_explicit_only,
    micro_prf,
    per_skill_prf,
    macro_f1,
)
from eval.error_analysis import collect_errors, save_errors_csv

USE_PANDAS = True
try:
    import pandas as pd
except Exception:
    USE_PANDAS = False

USE_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    USE_MPL = False


def normalize_rows(rows: List[Any]) -> List[Dict]:
    """
    Convert dataset rows into a uniform dict format:
    {"idx": int, "text": str, "skills": dict}

    Supports:
      - dict rows: {"text":..., "skills":..., "idx":...}
      - tuple rows: (text, skills) OR (idx, text, skills)
    """
    norm: List[Dict] = []
    for i, r in enumerate(rows, start=1):
        if isinstance(r, dict):
            norm.append(
                {
                    "idx": int(r.get("idx", i)),
                    "text": str(r.get("text", "")),
                    "skills": r.get("skills", {}) or {},
                }
            )
            continue

        if isinstance(r, (tuple, list)):
            if len(r) == 2:
                text, skills = r
                norm.append({"idx": i, "text": str(text), "skills": skills or {}})
                continue
            if len(r) == 3:
                idx, text, skills = r
                norm.append({"idx": int(idx), "text": str(text), "skills": skills or {}})
                continue

        raise TypeError(f"Unsupported row format at i={i}: type={type(r)} value={r}")

    return norm


def align_gt_to_global(gt: Dict[str, float]) -> Dict[str, float]:
    aligned = {s: 0.0 for s in GLOBAL_SKILL_VECTOR}
    for s, v in gt.items():
        if s in aligned:
            aligned[s] = float(v)
    return aligned


def save_per_skill_tables(out_dir: Path, per_skill: Dict[str, Dict[str, float]], prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    items_sorted = sorted(per_skill.items(), key=lambda kv: kv[1]["f1"], reverse=True)

    top10 = items_sorted[:10]
    bottom10 = list(reversed(items_sorted[-10:]))

    if USE_PANDAS:
        import pandas as pd

        pd.DataFrame([{"skill": s, **m} for s, m in items_sorted]).to_csv(
            out_dir / f"{prefix}_per_skill_all.csv", index=False
        )
        pd.DataFrame([{"skill": s, **m} for s, m in top10]).to_csv(
            out_dir / f"{prefix}_per_skill_top10.csv", index=False
        )
        pd.DataFrame([{"skill": s, **m} for s, m in bottom10]).to_csv(
            out_dir / f"{prefix}_per_skill_bottom10.csv", index=False
        )

    return top10, bottom10


def save_top_bottom_plot(out_dir: Path, items, title: str, filename: str):
    if not USE_MPL:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    skills = [s for s, _ in items]
    f1s = [m["f1"] for _, m in items]

    plt.figure(figsize=(10, 6))
    plt.barh(list(reversed(skills)), list(reversed(f1s)))
    plt.title(title)
    plt.xlabel("F1")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def evaluate_binary(y_true_float: List[Dict[str, float]], y_pred_float: List[Dict[str, float]], mode: str):
    if mode == "any":
        bin_fn = binarize_any_evidence
        title = "Any evidence (0.5 or 1.0 = positive)"
    elif mode == "explicit":
        bin_fn = binarize_explicit_only
        title = "Explicit only (1.0 = positive)"
    else:
        raise ValueError("mode must be 'any' or 'explicit'")

    y_true_flat, y_pred_flat = [], []
    y_true_by_skill = {s: [] for s in GLOBAL_SKILL_VECTOR}
    y_pred_by_skill = {s: [] for s in GLOBAL_SKILL_VECTOR}

    for gt, pr in zip(y_true_float, y_pred_float):
        for s in GLOBAL_SKILL_VECTOR:
            t = bin_fn(gt[s])
            p = bin_fn(pr[s])
            y_true_flat.append(t)
            y_pred_flat.append(p)
            y_true_by_skill[s].append(t)
            y_pred_by_skill[s].append(p)

    micro = micro_prf(y_true_flat, y_pred_flat)
    per_skill = per_skill_prf(y_true_by_skill, y_pred_by_skill)
    micro["macro_f1"] = macro_f1(per_skill)
    micro["mode"] = title
    return micro, per_skill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to trained HF model folder (e.g., outputs/.../final)")
    ap.add_argument("--data", default="data/splits/test.jsonl", help="Test split JSONL path")
    ap.add_argument("--out", default="outputs/eval_model", help="Output dir")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N samples (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for predicted positive")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
    import torch

    num_labels = len(GLOBAL_SKILL_VECTOR)

    config = AutoConfig.from_pretrained(str(model_dir))
    # Safety: force multi-label settings even if not saved correctly
    config.num_labels = num_labels
    config.problem_type = "multi_label_classification"

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), config=config)
    model.eval()

    # --- Load data ---
    rows_raw = load_jsonl_dataset(str(data_path))
    rows = normalize_rows(rows_raw)

    print("Loaded samples:", len(rows))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
        print("Using only first samples:", len(rows))

    y_true_float: List[Dict[str, float]] = []
    y_pred_float: List[Dict[str, float]] = []

    # Predict: multi-label sigmoid over logits
    with torch.no_grad():
        for r in rows:
            gt = align_gt_to_global(r["skills"])

            enc = tokenizer(
                r["text"],
                truncation=True,
                padding=True,
                max_length=256,
                return_tensors="pt",
            )

            logits = model(**enc).logits.squeeze(0)
            probs = torch.sigmoid(logits).cpu().tolist()  # length = num_labels

            pred = {
                GLOBAL_SKILL_VECTOR[i]: (1.0 if probs[i] >= args.threshold else 0.0)
                for i in range(num_labels)
            }

            y_true_float.append(gt)
            y_pred_float.append(pred)

    # Evaluate in the SAME two binary views (for consistency vs baselines)
    micro_any, per_skill_any = evaluate_binary(y_true_float, y_pred_float, mode="any")
    micro_exp, per_skill_exp = evaluate_binary(y_true_float, y_pred_float, mode="explicit")

    summary = [micro_any, micro_exp]
    if USE_PANDAS:
        import pandas as pd

        pd.DataFrame(summary).to_csv(out_dir / "overall_metrics.csv", index=False)

    print("\nOverall metrics:")
    for m in summary:
        print(
            f"- {m['mode']}: micro_f1={m['f1']:.4f}  macro_f1={m['macro_f1']:.4f}  "
            f"P={m['precision']:.4f} R={m['recall']:.4f}"
        )

    top_any, bot_any = save_per_skill_tables(out_dir, per_skill_any, prefix="any")
    top_exp, bot_exp = save_per_skill_tables(out_dir, per_skill_exp, prefix="explicit")

    save_top_bottom_plot(out_dir, top_any, "Top 10 skills by F1 (Any evidence)", "any_top10_f1.png")
    save_top_bottom_plot(out_dir, bot_any, "Bottom 10 skills by F1 (Any evidence)", "any_bottom10_f1.png")
    save_top_bottom_plot(out_dir, top_exp, "Top 10 skills by F1 (Explicit-only)", "explicit_top10_f1.png")
    save_top_bottom_plot(out_dir, bot_exp, "Bottom 10 skills by F1 (Explicit-only)", "explicit_bottom10_f1.png")

    # Error analysis
    errors_any = collect_errors(rows, y_true_float, y_pred_float, mode="any", limit_per_type=200)
    save_errors_csv(out_dir, errors_any, "errors_any_fp_fn.csv")

    errors_exp = collect_errors(rows, y_true_float, y_pred_float, mode="explicit", limit_per_type=200)
    save_errors_csv(out_dir, errors_exp, "errors_explicit_fp_fn.csv")

    print("\nSaved outputs to:", out_dir.resolve())
    print("Look at:")
    print("- overall_metrics.csv")
    print("- errors_any_fp_fn.csv / errors_explicit_fp_fn.csv")


if __name__ == "__main__":
    main()
