# src/datasets/synthetic_jsonl.py
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from skills.globalVector import GLOBAL_SKILL_VECTOR


def _extract_text(obj: Dict) -> str | None:
    return obj.get("job_description") or obj.get("resume_chunk_text") or obj.get("text")


def _align_labels_to_global(skills_dict: Dict[str, float]) -> List[float]:
    """
    Convert {skill: 0/0.5/1} into a dense list aligned to GLOBAL_SKILL_VECTOR order.
    Missing skills -> 0.0
    """
    aligned = [0.0] * len(GLOBAL_SKILL_VECTOR)
    if not isinstance(skills_dict, dict):
        return aligned

    idx_map = {s: i for i, s in enumerate(GLOBAL_SKILL_VECTOR)}
    for s, v in skills_dict.items():
        if s in idx_map:
            try:
                aligned[idx_map[s]] = float(v)
            except Exception:
                aligned[idx_map[s]] = 0.0
    return aligned


def load_jsonl_dataset(path: str | Path) -> List[Tuple[str, List[float]]]:
    """
    Returns list of (text, labels_vector) where labels_vector length == len(GLOBAL_SKILL_VECTOR)
    """
    path = Path(path)
    rows: List[Tuple[str, List[float]]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = _extract_text(obj)
            skills = obj.get("skills", {})

            if not text:
                continue

            labels = _align_labels_to_global(skills)
            rows.append((str(text), labels))

    return rows


def split_dataset(
    data: List[Tuple[str, List[float]]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Dict[str, List[Tuple[str, List[float]]]]:
    """
    Splits into train/val/test. test_ratio = 1 - train_ratio - val_ratio
    """
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1

    n = len(data)
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = [data[i] for i in idxs[:n_train]]
    val = [data[i] for i in idxs[n_train:n_train + n_val]]
    test = [data[i] for i in idxs[n_train + n_val:]]

    return {"train": train, "val": val, "test": test}
