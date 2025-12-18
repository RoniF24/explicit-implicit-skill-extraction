# eval/evaluate.py
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Make project root importable (so "skills/..." and "baselines/..." work)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR
from baselines.keyword_baseline import predict_keyword_baseline
from baselines.zero_shot_baseline import predict_zero_shot_baseline
from eval.metrics import (
    binarize_any_evidence,
    binarize_explicit_only,
    micro_prf,
    per_skill_prf,
    macro_f1,
)
from eval.error_analysis import collect_errors, save_errors_csv

# Optional pandas for saving CSV
USE_PANDAS = True
try:
    import pandas as pd
except Exception:
    USE_PANDAS = False

# Optional matplotlib for plots
USE_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    USE_MPL = False


def load_jsonl(jsonl_path: Path) -> List[Dict]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("job_description") or obj.get("resume_chunk_text") or obj.get("text")
            skills = obj.get("skills", {})

            if text is None or not isinstance(skills, dict):
                continue

            rows.append(
                {
                    "idx": i,
                    "text": str(text),
                    "skills": skills,  # ground truth: skill -> 0/0.5/1
                }
            )
    return rows


def align_gt_to_global(gt: Dict[str, float]) -> Dict[str, float]:
    """
    Ensure GT has ALL skills from GLOBAL_SKILL_VECTOR.
    Missing skills are treated as 0.0.
    """
    aligned = {s: 0.0 for s in GLOBAL_SKILL_VECTOR}
    for s, v in gt.items():
        if s in aligned:
            aligned[s] = float(v)
    return aligned


def get_predictor(name: str):
    name = name.lower().strip()
    if name == "keyword":
        return predict_keyword_baseline
    if name == "zero_shot":
        return predict_zero_shot_baseline
    raise ValueError(f"Unknown baseline: {name}. Supported: keyword, zero_shot")


def evaluate_binary(
    rows: List[Dict],
    y_true_float: List[Dict[str, float]],
    y_pred_float: List[Dict[str, float]],
    mode: str,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    mode:
      - "any": (0.5 or 1.0) counts as positive
      - "explicit": only 1.0 counts as positive
    """
    if mode == "any":
        bin_fn = binarize_any_evidence
        title = "Any evidence (0.5 or 1.0 = positive)"
    elif mode == "explicit":
        bin_fn = binarize_explicit_only
        title = "Explicit only (1.0 = positive)"
    else:
        raise ValueError("mode must be 'any' or 'explicit'")

    y_true_flat: List[int] = []
    y_pred_flat: List[int] = []

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


def save_per_skill_tables(out_dir: Path, per_skill: Dict[str, Dict[str, float]], prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    items = list(per_skill.items())
    items_sorted = sorted(items, key=lambda kv: kv[1]["f1"], reverse=True)

    top10 = items_sorted[:10]
    bottom10 = list(reversed(items_sorted[-10:]))

    if USE_PANDAS:
        import pandas as pd

        df_all = pd.DataFrame([{"skill": s, **m} for s, m in items_sorted])
        df_all.to_csv(out_dir / f"{prefix}_per_skill_all.csv", index=False)

        pd.DataFrame([{"skill": s, **m} for s, m in top10]).to_csv(out_dir / f"{prefix}_per_skill_top10.csv", index=False)
        pd.DataFrame([{"skill": s, **m} for s, m in bottom10]).to_csv(out_dir / f"{prefix}_per_skill_bottom10.csv", index=False)

    return top10, bottom10


def save_top_bottom_plot(out_dir: Path, items: List[Tuple[str, Dict[str, float]]], title: str, filename: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="keyword", choices=["keyword", "zero_shot"])
    ap.add_argument("--data", default="data/synthetic_dataset.jsonl")
    ap.add_argument("--out", default="outputs/eval_keyword")

    # NEW: run only first N samples (useful for slow baselines like zero_shot)
    # Usage examples:
    #   --limit 10   -> run first 10 samples (quick sanity check)
    #   --limit 100  -> run first 100 samples
    #   --limit 0    -> run ALL samples (default)
    ap.add_argument("--limit", type=int, default=0, help="Run only first N samples (0 = all)")

    args = ap.parse_args()

    dataset_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(dataset_path)
    print("Loaded samples:", len(rows))

    # NEW: apply limit after loading
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
        print("Using only first samples:", len(rows))

    predictor = get_predictor(args.baseline)

    y_true_float: List[Dict[str, float]] = []
    y_pred_float: List[Dict[str, float]] = []

    for r in rows:
        gt = align_gt_to_global(r["skills"])
        pr = predictor(r["text"])

        for s in GLOBAL_SKILL_VECTOR:
            pr.setdefault(s, 0.0)

        y_true_float.append(gt)
        y_pred_float.append(pr)

    micro_any, per_skill_any = evaluate_binary(rows, y_true_float, y_pred_float, mode="any")
    micro_exp, per_skill_exp = evaluate_binary(rows, y_true_float, y_pred_float, mode="explicit")

    summary = [micro_any, micro_exp]
    if USE_PANDAS:
        import pandas as pd
        pd.DataFrame(summary).to_csv(out_dir / "overall_metrics.csv", index=False)

    print("\nOverall metrics:")
    for m in summary:
        print(f"- {m['mode']}: micro_f1={m['f1']:.4f}  macro_f1={m['macro_f1']:.4f}  P={m['precision']:.4f} R={m['recall']:.4f}")

    top_any, bot_any = save_per_skill_tables(out_dir, per_skill_any, prefix="any")
    top_exp, bot_exp = save_per_skill_tables(out_dir, per_skill_exp, prefix="explicit")

    save_top_bottom_plot(out_dir, top_any, "Top 10 skills by F1 (Any evidence)", "any_top10_f1.png")
    save_top_bottom_plot(out_dir, bot_any, "Bottom 10 skills by F1 (Any evidence)", "any_bottom10_f1.png")
    save_top_bottom_plot(out_dir, top_exp, "Top 10 skills by F1 (Explicit-only)", "explicit_top10_f1.png")
    save_top_bottom_plot(out_dir, bot_exp, "Bottom 10 skills by F1 (Explicit-only)", "explicit_bottom10_f1.png")

    errors_any = collect_errors(rows, y_true_float, y_pred_float, mode="any", limit_per_type=200)
    save_errors_csv(out_dir, errors_any, "errors_any_fp_fn.csv")

    errors_exp = collect_errors(rows, y_true_float, y_pred_float, mode="explicit", limit_per_type=200)
    save_errors_csv(out_dir, errors_exp, "errors_explicit_fp_fn.csv")

    print("\nSaved outputs to:", out_dir.resolve())
    print("Files you should look at:")
    print("- overall_metrics.csv")
    print("- any_per_skill_top10.csv / any_per_skill_bottom10.csv")
    print("- explicit_per_skill_top10.csv / explicit_per_skill_bottom10.csv")
    print("- errors_any_fp_fn.csv / errors_explicit_fp_fn.csv")


if __name__ == "__main__":
    main()
