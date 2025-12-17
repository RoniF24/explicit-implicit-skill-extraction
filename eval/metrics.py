# eval/metrics.py
from __future__ import annotations

from typing import Dict, List, Tuple


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def binarize_any_evidence(y: float) -> int:
    """
    For "Any evidence" evaluation:
    - treat 0.5 and 1.0 as positive
    - treat 0.0 as negative
    """
    return 1 if y > 0.0 else 0


def binarize_explicit_only(y: float) -> int:
    """
    For "Explicit-only" evaluation:
    - 1.0 is positive
    - 0.5, 0.0 are negative
    """
    return 1 if y == 1.0 else 0


def micro_prf(y_true_bin: List[int], y_pred_bin: List[int]) -> Dict[str, float]:
    """
    Micro-averaged P/R/F1 over flattened binary labels.
    """
    tp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true_bin, y_pred_bin) if t == 1 and p == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def per_skill_prf(
    y_true_by_skill: Dict[str, List[int]],
    y_pred_by_skill: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """
    Per-skill P/R/F1 (binary) + supports macro averaging later.
    """
    out: Dict[str, Dict[str, float]] = {}

    for skill, t_list in y_true_by_skill.items():
        p_list = y_pred_by_skill[skill]

        tp = sum(1 for t, p in zip(t_list, p_list) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(t_list, p_list) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(t_list, p_list) if t == 1 and p == 0)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        support_pos = sum(t_list)

        out[skill] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support_pos": float(support_pos),
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
        }

    return out


def macro_f1(per_skill: Dict[str, Dict[str, float]]) -> float:
    if not per_skill:
        return 0.0
    return sum(v["f1"] for v in per_skill.values()) / len(per_skill)
