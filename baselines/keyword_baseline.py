# baselines/keyword_baseline.py
from __future__ import annotations

import os
import sys
import re
from typing import Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.skillAliases import skills as SKILL_ALIASES
from skills.globalVector import GLOBAL_SKILL_VECTOR


def _normalize_text(text: str) -> str:
    return (text or "").strip()


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    phrase = (phrase or "").strip()
    esc = re.escape(phrase)
    return re.compile(rf"(?<!\w){esc}(?!\w)", re.IGNORECASE)


def build_alias_patterns() -> Dict[str, List[re.Pattern]]:
    patterns: Dict[str, List[re.Pattern]] = {}
    for canon, meta in SKILL_ALIASES.items():
        items = [canon] + list(meta.get("aliases", []))

        filtered: List[str] = []
        for a in items:
            a = (a or "").strip()
            if len(a) < 2:
                continue
            filtered.append(a)

        patterns[canon] = [_word_boundary_pattern(x) for x in filtered]
    return patterns


_ALIAS_PATTERNS = build_alias_patterns()


def predict_keyword_baseline(text: str) -> Dict[str, float]:
    """
    Explicit-only baseline:
    - If canonical/alias appears -> 1.0
    - Else -> 0.0
    Returns FULL dict over GLOBAL_SKILL_VECTOR.
    """
    t = _normalize_text(text)
    pred: Dict[str, float] = {s: 0.0 for s in GLOBAL_SKILL_VECTOR}
    if not t:
        return pred

    for canon in GLOBAL_SKILL_VECTOR:
        pats = _ALIAS_PATTERNS.get(canon, [])
        if any(p.search(t) for p in pats):
            pred[canon] = 1.0
    return pred


if __name__ == "__main__":
    sample = "Built REST APIs with Django and deployed with Docker. Used PostgreSQL and Redis."
    out = predict_keyword_baseline(sample)
    hits = {k: v for k, v in out.items() if v > 0}
    print(hits)
