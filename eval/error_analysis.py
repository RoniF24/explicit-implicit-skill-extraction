# eval/error_analysis.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR

USE_PANDAS = True
try:
    import pandas as pd
except Exception:
    USE_PANDAS = False


def _label_name(v: float) -> str:
    if v == 1.0:
        return "EXPLICIT"
    if v == 0.5:
        return "IMPLICIT"
    return "NONE"


def collect_errors(
    rows: List[Dict[str, Any]],
    y_true_float: List[Dict[str, float]],
    y_pred_float: List[Dict[str, float]],
    mode: str = "any",
    limit_per_type: int = 200,
) -> List[Dict[str, Any]]:
    """
    Collect error examples for inspection.

    mode:
      - "any": treat (0.5 or 1.0) as positive, 0.0 as negative
      - "explicit": treat 1.0 as positive, (0.5 or 0.0) as negative

    We also keep the original float labels (0/0.5/1) for explanation.
    """
    assert mode in ("any", "explicit")

    errors: List[Dict[str, Any]] = []
    counters = {"FP": 0, "FN": 0}

    def is_pos_any(x: float) -> bool:
        return x > 0.0

    def is_pos_exp(x: float) -> bool:
        return x == 1.0

    is_pos = is_pos_any if mode == "any" else is_pos_exp

    for r, gt, pr in zip(rows, y_true_float, y_pred_float):
        text = r["text"]

        for skill in GLOBAL_SKILL_VECTOR:
            t = float(gt[skill])
            p = float(pr[skill])

            t_pos = is_pos(t)
            p_pos = is_pos(p)

            # FP / FN
            if (not t_pos) and p_pos:
                if counters["FP"] < limit_per_type:
                    errors.append(
                        {
                            "error_type": "FP",
                            "mode": mode,
                            "idx": r["idx"],
                            "skill": skill,
                            "gt_value": t,
                            "gt_label": _label_name(t),
                            "pred_value": p,
                            "pred_label": _label_name(p),
                            "text_snippet": text[:260].replace("\n", " "),
                        }
                    )
                counters["FP"] += 1

            elif t_pos and (not p_pos):
                if counters["FN"] < limit_per_type:
                    errors.append(
                        {
                            "error_type": "FN",
                            "mode": mode,
                            "idx": r["idx"],
                            "skill": skill,
                            "gt_value": t,
                            "gt_label": _label_name(t),
                            "pred_value": p,
                            "pred_label": _label_name(p),
                            "text_snippet": text[:260].replace("\n", " "),
                        }
                    )
                counters["FN"] += 1

    return errors


def save_errors_csv(out_dir: Path, errors: List[Dict[str, Any]], filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    if USE_PANDAS:
        pd.DataFrame(errors).to_csv(out_dir / filename, index=False)
    else:
        # fallback: write TSV
        import csv
        with (out_dir / filename.replace(".csv", ".tsv")).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(errors[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(errors)


def summarize_error_counts(errors: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    """
    Count errors by (mode, error_type)
    """
    out: Dict[Tuple[str, str], int] = {}
    for e in errors:
        key = (e["mode"], e["error_type"])
        out[key] = out.get(key, 0) + 1
    return out
