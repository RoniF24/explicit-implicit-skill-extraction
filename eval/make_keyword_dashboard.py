from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib is required. Install: pip install matplotlib") from e


# ---------------- CONFIG ----------------
MIN_SUPPORT_EXPLICIT = 5


# ---------------- IO HELPERS ----------------
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows


def parse_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def count_jsonl_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def infer_dataset_name(run_dir: Path) -> str:
    for p in run_dir.parts:
        if p.startswith("dataset__"):
            return p.replace("dataset__", "")
    return run_dir.parent.name


# ---------------- LOAD METRICS ----------------
def load_overall_explicit(run_dir: Path) -> Optional[Dict[str, float]]:
    rows = read_csv_rows(run_dir / "overall_metrics.csv")
    for r in rows:
        mode = (r.get("mode") or "").lower()
        if ("explicit" in mode) or (">=1.0" in mode) or ("explicit_only" in mode) or ("1.0_only" in mode) or ("==1.0" in mode):
            return {
                "precision": parse_float(r.get("precision"), 0.0),
                "recall": parse_float(r.get("recall"), 0.0),
                "f1": parse_float(r.get("f1"), 0.0),
                "tp": parse_float(r.get("tp"), 0.0),
                "fp": parse_float(r.get("fp"), 0.0),
                "fn": parse_float(r.get("fn"), 0.0),
            }
    return None


def load_per_skill_explicit(run_dir: Path) -> Tuple[List[Tuple[str, float, int]], bool]:
    rows = read_csv_rows(run_dir / "per_skill__explicit_only__f1.csv")
    items: List[Tuple[str, float, int]] = []

    has_support_col = False
    if rows:
        has_support_col = ("support" in rows[0])

    for r in rows:
        s = (r.get("skill") or "").strip()
        if not s:
            continue
        f1 = parse_float(r.get("f1"), 0.0)
        sup = int(parse_float(r.get("support"), 0.0)) if has_support_col else 0
        items.append((s, f1, sup))

    return items, has_support_col


def count_errors_explicit(run_dir: Path) -> Tuple[int, int]:
    rows = read_csv_rows(run_dir / "errors__explicit_only__fp_fn.csv")
    fp = sum(1 for r in rows if (r.get("type") or "").upper() == "FP")
    fn = sum(1 for r in rows if (r.get("type") or "").upper() == "FN")
    return fp, fn


def top_bottom(
    items: List[Tuple[str, float, int]],
    k: int = 5,
    *,
    min_support: int = MIN_SUPPORT_EXPLICIT,
    use_support_filter: bool = True,
) -> Tuple[List[Tuple[str, float, int]], List[Tuple[str, float, int]]]:
    if not items:
        return [], []

    filtered = items
    if use_support_filter:
        filtered = [x for x in items if x[2] >= min_support]

    if not filtered:
        return [], []

    items_desc = sorted(filtered, key=lambda x: x[1], reverse=True)
    topk = items_desc[:k]
    bottomk = sorted(items_desc, key=lambda x: x[1])[:k]
    return topk, bottomk


# ---------------- DASHBOARD ----------------
def make_keyword_dashboard(run_dir: Path, out_name: str) -> Path:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    dataset = infer_dataset_name(run_dir)
    samples = count_jsonl_lines(run_dir / "preds.jsonl")

    exp = load_overall_explicit(run_dir)
    per_exp, has_support = load_per_skill_explicit(run_dir)

    top5, bot5 = top_bottom(
        per_exp,
        5,
        min_support=MIN_SUPPORT_EXPLICIT,
        use_support_filter=has_support,
    )

    fp, fn = count_errors_explicit(run_dir)

    # ---- FIGURE: keep baseline/eval text, remove right-side error examples ----
    fig = plt.figure(figsize=(18, 10), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.15, 1.85])

    ax_text = fig.add_subplot(gs[0, 0])
    ax_micro = fig.add_subplot(gs[0, 1])
    ax_top = fig.add_subplot(gs[1, 0])
    ax_bottom = fig.add_subplot(gs[1, 1])

    fig.suptitle(f"Dashboard — keyword_matching | dataset: {dataset}", fontsize=22, fontweight="bold", y=0.98)

    # Summary (kept)
    ax_text.axis("off")
    lines = [
        "Baseline",
        "- Type: keyword matching (aliases + word-boundary regex)",
        "- Output: Explicit only (1.0 if match else 0.0)",
        "",
        "Evaluation (Explicit only, >=1.0 positive)",
        f"- Run dir: {run_dir.name}",
        f"- Samples evaluated: {samples}",
    ]
    if exp:
        lines += [
            f"- Precision: {exp['precision']:.3f}    Recall: {exp['recall']:.3f}    F1: {exp['f1']:.3f}",
            f"- TP/FP/FN: {int(exp['tp'])}/{int(exp['fp'])}/{int(exp['fn'])}",
        ]
    else:
        lines += ["- (Missing explicit metrics in overall_metrics.csv)"]

    lines += [f"- Errors: FP={fp} | FN={fn}"]

    if has_support:
        lines += [f"- Top/Bottom filter: support >= {MIN_SUPPORT_EXPLICIT} (explicit positives)"]
    else:
        lines += ["- Top/Bottom filter: support not available (raw F1)"]

    ax_text.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=11)

    # Micro metrics
    ax_micro.set_title("Micro metrics (Explicit only)", fontsize=16, fontweight="bold")
    if not exp:
        ax_micro.text(0.5, 0.5, "No explicit metrics found", ha="center", va="center")
        ax_micro.set_xticks([])
        ax_micro.set_yticks([])
    else:
        labels = ["PRECISION", "RECALL", "F1"]
        vals = [exp["precision"], exp["recall"], exp["f1"]]
        ax_micro.bar([0, 1, 2], vals, width=0.55)
        ax_micro.set_ylim(0.0, 1.05)
        ax_micro.set_xticks([0, 1, 2])
        ax_micro.set_xticklabels(labels)

    # Top explicit
    ax_top.set_title("Top 5 skills by F1 (Explicit)", fontsize=16, fontweight="bold")
    if not top5:
        ax_top.axis("off")
        ax_top.text(0.5, 0.5, f"No skills pass support >= {MIN_SUPPORT_EXPLICIT}", ha="center", va="center")
    else:
        sk = [s for s, _, _ in top5]
        f1s = [v for _, v, _ in top5]
        ax_top.barh(list(reversed(sk)), list(reversed(f1s)))
        ax_top.set_xlim(0.0, 1.05)
        ax_top.set_xlabel("F1")

    # Bottom explicit
    ax_bottom.set_title("Bottom 5 skills by F1 (Explicit)", fontsize=16, fontweight="bold")
    if not bot5:
        ax_bottom.axis("off")
        ax_bottom.text(0.5, 0.5, f"No skills pass support >= {MIN_SUPPORT_EXPLICIT}", ha="center", va="center")
    else:
        sk = [s for s, _, _ in bot5]
        f1s = [v for _, v, _ in bot5]
        ax_bottom.barh(list(reversed(sk)), list(reversed(f1s)))
        ax_bottom.set_xlim(0.0, 1.05)
        ax_bottom.set_xlabel("F1")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = run_dir / out_name
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to run__latest directory")
    ap.add_argument("--out_name", default="dashboard__keyword__explicit.png", help="Output file name (inside run_dir)")
    args = ap.parse_args()

    out = make_keyword_dashboard(Path(args.run_dir), args.out_name)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()
