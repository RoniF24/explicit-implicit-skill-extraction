# src/datasets/synthetic_jsonl.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skills.globalVector import GLOBAL_SKILL_VECTOR


def _extract_text(obj: Dict[str, Any]) -> Optional[str]:
    return obj.get("text") or obj.get("job_description") or obj.get("resume_chunk_text")


def align_skills_to_global_vector(skills_dict: Dict[str, float]) -> List[float]:
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


def load_jsonl_rows(path: str | Path) -> List[Dict[str, Any]]:
    """
    Standard loader for evaluation / error analysis:
      { "idx": int, "text": str, "skills": Dict[str,float] }
    """
    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = _extract_text(obj)
            skills = obj.get("skills", {})

            if not text or not isinstance(skills, dict):
                continue

            rows.append({"idx": i, "text": str(text), "skills": {k: float(v) for k, v in skills.items()}})

    return rows


def load_jsonl_dataset(path: str | Path) -> List[Tuple[str, List[float]]]:
    """
    Training-friendly loader:
      returns list of (text, labels_vector) aligned to GLOBAL_SKILL_VECTOR
    """
    rows = load_jsonl_rows(path)
    out: List[Tuple[str, List[float]]] = []
    for r in rows:
        out.append((r["text"], align_skills_to_global_vector(r["skills"])))
    return out
