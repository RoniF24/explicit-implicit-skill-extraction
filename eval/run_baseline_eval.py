# eval/run_baseline_eval.py
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR

# Baselines
from baselines.keyword_baseline import predict_keyword_baseline
from baselines.zero_shot_baseline import predict_from_text as predict_zero_shot_baseline  # unused if baseline=keyword

# --------- Optional deps for pretty outputs ----------
USE_PANDAS = True
try:
    import pandas as pd  # noqa: F401
except Exception:
    USE_PANDAS = False

USE_MPL = True
try:
    import matplotlib.pyplot as plt  # noqa: F401
except Exception:
    USE_MPL = False

# --------- Progress bar ----------
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

EPS = 1e-9


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("job_description") or obj.get("resume_chunk_text") or obj.get("text") or ""
            skills = obj.get("skills", {}) if isinstance(obj.get("skills", {}), dict) else {}
            rows.append({"idx": i, "text": str(text), "skills": skills})
    return rows


def align_to_global(d: Dict[str, float]) -> Dict[str, float]:
    out = {s: 0.0 for s in GLOBAL_SKILL_VECTOR}
    for k, v in (d or {}).items():
        if k in out:
            try:
                out[k] = float(v)
            except Exception:
                out[k] = 0.0
    return out


# ---------------- Binarizers ----------------
def bin_any(v: float) -> int:
    return 1 if float(v) >= 0.5 else 0


def bin_explicit(v: float) -> int:
    return 1 if float(v) >= 1.0 else 0


def _is_half(v: float) -> bool:
    try:
        return abs(float(v) - 0.5) <= EPS
    except Exception:
        return False


def bin_implicit_only(v: float) -> int:
    return 1 if _is_half(v) else 0


# ---------------- Metrics ----------------
def micro_prf(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def per_skill_f1(y_true_by_skill: Dict[str, List[int]], y_pred_by_skill: Dict[str, List[int]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in GLOBAL_SKILL_VECTOR:
        m = micro_prf(y_true_by_skill[s], y_pred_by_skill[s])
        out[s] = m["f1"]
    return out


def _bin_fn_for_mode(mode: str):
    m = mode.lower().strip()
    if m == "any":
        return bin_any
    if m == "explicit":
        return bin_explicit
    if m == "implicit":
        return bin_implicit_only
    raise ValueError(f"Unknown mode: {mode}. Expected: any / explicit / implicit")


def collect_errors(
    rows: List[Dict[str, Any]],
    y_true: List[Dict[str, float]],
    y_pred: List[Dict[str, float]],
    mode: str,
    limit: int = 200,
):
    bin_fn = _bin_fn_for_mode(mode)
    errors = []
    fp_count = 0
    fn_count = 0

    for r, gt, pr in zip(rows, y_true, y_pred):
        for s in GLOBAL_SKILL_VECTOR:
            t = bin_fn(gt[s])
            p = bin_fn(pr[s])
            if t == 0 and p == 1 and fp_count < limit:
                errors.append({"type": "FP", "mode": mode, "idx": r["idx"], "skill": s, "gt": gt[s], "pred": pr[s], "text": r["text"]})
                fp_count += 1
            if t == 1 and p == 0 and fn_count < limit:
                errors.append({"type": "FN", "mode": mode, "idx": r["idx"], "skill": s, "gt": gt[s], "pred": pr[s], "text": r["text"]})
                fn_count += 1
    return errors


def save_csv(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if USE_PANDAS and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = str(r.get(c, "")).replace("\n", " ").replace("\r", " ").replace(",", ";")
            vals.append(v)
        lines.append(",".join(vals))
    path.write_text("\n".join(lines), encoding="utf-8")


def save_bar_plot(path: Path, labels: List[str], values: List[float], title: str, xlabel: str):
    if not USE_MPL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("F1")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_barh_plot(path: Path, labels: List[str], values: List[float], title: str):
    if not USE_MPL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.barh(list(reversed(labels)), list(reversed(values)))
    plt.title(title)
    plt.xlabel("F1")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def get_predictor(baseline: str):
    b = baseline.lower().strip()
    if b in ("keyword", "keyword_matching"):
        return predict_keyword_baseline
    if b == "zero_shot":
        return predict_zero_shot_baseline
    raise ValueError("baseline must be: keyword / keyword_matching / zero_shot")


def _baseline_run_id(args_run_name: str) -> str:
    rn = (args_run_name or "").strip()
    if rn:
        return rn
    return f"run__{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _make_base_id_for_zero_shot() -> str:
    backend = os.getenv("ZERO_SHOT_BACKEND", "ollama").strip().lower()
    temp = os.getenv("ZERO_SHOT_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0"))
    if backend == "ollama":
        model = os.getenv("OLLAMA_MODEL", "unknown_model")
        safe_model = model.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        return f"zero_shot__ollama__{safe_model}__temp{temp}"
    else:
        model = os.getenv("OPENAI_MODEL", "unknown_model")
        safe_model = model.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        return f"zero_shot__openai__{safe_model}__temp{temp}"


def run_one_baseline(
    baseline_name: str,
    rows: List[Dict[str, Any]],
    dataset_path: Path,
    out_root: str,
    tag: str,
    run_name: str,
) -> Path:
    predictor = get_predictor(baseline_name)
    dataset_stem = dataset_path.stem

    if baseline_name == "zero_shot":
        base_id = _make_base_id_for_zero_shot()
    else:
        base_id = "keyword_matching__explicit_only"

    if tag.strip():
        base_id = f"{base_id}__{tag.strip()}"

    out_dir = Path(out_root) / baseline_name / base_id / f"dataset__{dataset_stem}" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Predict -----
    y_true_float: List[Dict[str, float]] = []
    y_pred_float: List[Dict[str, float]] = []
    preds_jsonl_path = out_dir / "preds.jsonl"

    iter_rows = tqdm(rows, desc=f"Predicting ({baseline_name})", unit="sample") if tqdm else rows

    with preds_jsonl_path.open("w", encoding="utf-8") as wf:
        for r in iter_rows:
            gt = align_to_global(r["skills"])

            if baseline_name == "zero_shot":
                pred_obj = predictor(idx=r["idx"], job_description=r["text"])
                pred_sparse = pred_obj.get("skills", {}) if isinstance(pred_obj, dict) else {}
            else:
                pred_sparse = predictor(r["text"])

            pr = align_to_global(pred_sparse)

            y_true_float.append(gt)
            y_pred_float.append(pr)

            wf.write(json.dumps({"idx": r["idx"], "skills": pred_sparse}, ensure_ascii=False) + "\n")

    # ----- Evaluate modes: any / explicit / implicit -----
    def eval_mode(mode: str) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        bin_fn = _bin_fn_for_mode(mode)

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
        per_skill = per_skill_f1(y_true_by_skill, y_pred_by_skill)
        support = {s: int(sum(y_true_by_skill[s])) for s in GLOBAL_SKILL_VECTOR}
        return micro, per_skill, support

    micro_any, f1_any, sup_any = eval_mode("any")
    micro_exp, f1_exp, sup_exp = eval_mode("explicit")
    micro_imp, f1_imp, sup_imp = eval_mode("implicit")

    # ----- Save overall metrics -----
    overall = [
        {"mode": "explicit_only(1.0_positive)", **micro_exp},
        {"mode": "implicit_only(0.5_positive)", **micro_imp},
        {"mode": "any_evidence(0.5_or_1.0_positive)", **micro_any},
    ]
    save_csv(out_dir / "overall_metrics.csv", overall)

    # ----- Save per-skill tables (WITH support) -----
    per_any_rows = [{"skill": s, "f1": f1_any[s], "support": sup_any[s]} for s in GLOBAL_SKILL_VECTOR]
    per_exp_rows = [{"skill": s, "f1": f1_exp[s], "support": sup_exp[s]} for s in GLOBAL_SKILL_VECTOR]
    per_imp_rows = [{"skill": s, "f1": f1_imp[s], "support": sup_imp[s]} for s in GLOBAL_SKILL_VECTOR]

    per_any_rows_sorted = sorted(per_any_rows, key=lambda x: x["f1"], reverse=True)
    per_exp_rows_sorted = sorted(per_exp_rows, key=lambda x: x["f1"], reverse=True)
    per_imp_rows_sorted = sorted(per_imp_rows, key=lambda x: x["f1"], reverse=True)

    save_csv(out_dir / "per_skill__any_evidence__f1.csv", per_any_rows_sorted)
    save_csv(out_dir / "per_skill__explicit_only__f1.csv", per_exp_rows_sorted)
    save_csv(out_dir / "per_skill__implicit_only__f1.csv", per_imp_rows_sorted)

    # ----- Plots -----
    save_bar_plot(
        out_dir / "plot__micro_f1__any_vs_explicit.png",
        ["any_evidence", "explicit_only"],
        [micro_any["f1"], micro_exp["f1"]],
        title=f"Micro-F1 ({baseline_name})",
        xlabel="Evaluation mode",
    )
    save_bar_plot(
        out_dir / "plot__micro_f1__explicit_vs_implicit_vs_any.png",
        ["explicit_only", "implicit_only", "any_evidence"],
        [micro_exp["f1"], micro_imp["f1"], micro_any["f1"]],
        title=f"Micro-F1 (3 modes) - {baseline_name}",
        xlabel="Evaluation mode",
    )

    top10_any = per_any_rows_sorted[:10]
    bot10_any = list(reversed(per_any_rows_sorted[-10:]))

    save_barh_plot(
        out_dir / "plot__top10__per_skill_f1__any_evidence.png",
        [x["skill"] for x in top10_any],
        [x["f1"] for x in top10_any],
        title=f"Top 10 skills by F1 (Any evidence) - {baseline_name}",
    )
    save_barh_plot(
        out_dir / "plot__bottom10__per_skill_f1__any_evidence.png",
        [x["skill"] for x in bot10_any],
        [x["f1"] for x in bot10_any],
        title=f"Bottom 10 skills by F1 (Any evidence) - {baseline_name}",
    )

    # ----- Error CSVs -----
    errors_any = collect_errors(rows, y_true_float, y_pred_float, mode="any", limit=200)
    errors_exp = collect_errors(rows, y_true_float, y_pred_float, mode="explicit", limit=200)
    errors_imp = collect_errors(rows, y_true_float, y_pred_float, mode="implicit", limit=200)

    save_csv(out_dir / "errors__any_evidence__fp_fn.csv", errors_any)
    save_csv(out_dir / "errors__explicit_only__fp_fn.csv", errors_exp)
    save_csv(out_dir / "errors__implicit_only__fp_fn.csv", errors_imp)

    print("\nDONE. Outputs saved to:")
    print(out_dir.resolve())
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", choices=["keyword", "zero_shot", "both"], required=True)
    ap.add_argument("--data", required=True, help="Path to JSONL dataset (with job_description + skills)")
    ap.add_argument("--out_root", default="outputs/validation", help="Root outputs folder")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N samples (0 = all)")
    ap.add_argument("--tag", default="", help="Optional extra tag for naming (e.g., aliases_v1)")
    ap.add_argument("--run_name", default="run__latest", help="Folder name for the run. Use run__latest to overwrite.")
    args = ap.parse_args()

    dataset_path = Path(args.data)
    rows = load_jsonl(dataset_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    run_name = _baseline_run_id(args.run_name)

    if args.baseline == "both":
        run_one_baseline("keyword", rows, dataset_path, args.out_root, args.tag, run_name)
        run_one_baseline("zero_shot", rows, dataset_path, args.out_root, args.tag, run_name)
    else:
        run_one_baseline(args.baseline, rows, dataset_path, args.out_root, args.tag, run_name)


if __name__ == "__main__":
    main()
