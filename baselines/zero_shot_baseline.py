# baselines/zero_shot_baseline.py
from __future__ import annotations

import os
import sys
import json
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.skillAliases import skills as SKILL_ALIASES
from skills.globalVector import GLOBAL_SKILL_VECTOR


# ---------------- Ollama config ----------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")  # change if needed
TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

# Cache to avoid re-calling for same text
CACHE_PATH = Path(os.getenv("ZERO_SHOT_CACHE", "outputs/cache/zero_shot_ollama_cache.json"))
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _normalize_text(t: str) -> str:
    return (t or "").strip()


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    # Match phrase with token boundaries (case-insensitive)
    esc = re.escape(phrase.lower())
    return re.compile(rf"(?<!\w){esc}(?!\w)")


def _build_alias_patterns() -> Dict[str, List[re.Pattern]]:
    """
    For each canonical skill, build patterns for canonical + aliases.
    We skip very short aliases (len < 2) to reduce false positives.
    """
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


_ALIAS_PATTERNS = _build_alias_patterns()


def extract_candidates_from_text(text: str, max_candidates: int = 30) -> List[str]:
    """
    Build a SHORTLIST from the text only (no GT leakage):
    candidates = skills whose canonical/alias appears in the text.

    This is the shortlist we send to the LLM.
    """
    t = (text or "").lower()
    hits: List[str] = []
    for skill, pats in _ALIAS_PATTERNS.items():
        for p in pats:
            if p.search(t):
                hits.append(skill)
                break

    # stable order: by appearance count is expensive; just sort
    hits = sorted(set(hits))
    return hits[:max_candidates]


def _hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_json_object(s: str) -> Optional[dict]:
    """
    Try to find and parse the first JSON object in model output.
    """
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = s[start : end + 1]
    try:
        return json.loads(chunk)
    except Exception:
        return None


def _ollama_generate(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "") or ""


def _build_prompt(text: str, candidates: List[str]) -> str:
    """
    We ask Ollama to label ONLY the provided candidate skills.
    Output must be JSON only.
    """
    cand_str = ", ".join(candidates)

    return f"""
You are doing skill evidence labeling for a CLOSED set of candidate skills.

LABELS (strict):
- 1.0 (EXPLICIT): the skill name appears clearly in the text (or obvious direct mention).
- 0.5 (IMPLICIT): the skill is strongly implied by responsibilities/processes, even if the name does NOT appear.
- 0.0 (NONE): not supported.

TASK:
Given the job description text, output labels ONLY for the candidate skills list provided below.

IMPORTANT RULES:
- Only use skills from the candidate list.
- Do NOT invent new skills.
- Output JSON ONLY. No extra text.

CANDIDATE SKILLS:
[{cand_str}]

JOB DESCRIPTION:
\"\"\"{text}\"\"\"

Return exactly this JSON schema:
{{
  "predicted_skills": {{
    "Skill A from candidate list": 0.0 or 0.5 or 1.0,
    "Skill B from candidate list": 0.0 or 0.5 or 1.0
  }}
}}
""".strip()


def predict_zero_shot_baseline(text: str, max_candidates: int = 30) -> Dict[str, float]:
    """
    Returns FULL dict over GLOBAL_SKILL_VECTOR with values in {0.0, 0.5, 1.0}.
    We only ask the LLM about a shortlist (from text), then fill the rest with 0.0.
    """
    text = _normalize_text(text)
    out: Dict[str, float] = {s: 0.0 for s in GLOBAL_SKILL_VECTOR}

    if not text:
        return out

    # shortlist candidates based on matches in the text (no GT leakage)
    candidates = extract_candidates_from_text(text, max_candidates=max_candidates)

    # If no candidates were found, nothing to label
    if not candidates:
        return out

    key = _hash_text(text) + f":{max_candidates}"
    cache = _load_cache()
    if key in cache:
        cached_pred = cache[key]
        # cached_pred is already a dict of skill->value
        for s, v in cached_pred.items():
            if s in out:
                out[s] = float(v)
        return out

    prompt = _build_prompt(text, candidates)
    raw = _ollama_generate(prompt)

    parsed = _extract_json_object(raw)
    if not parsed or "predicted_skills" not in parsed or not isinstance(parsed["predicted_skills"], dict):
        # fallback: if parsing fails, return all zeros for safety
        cache[key] = {}
        _save_cache(cache)
        return out

    pred_small: Dict[str, float] = {}
    for s, v in parsed["predicted_skills"].items():
        if s not in candidates:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        # clamp to allowed set
        if fv not in (0.0, 0.5, 1.0):
            # small tolerance
            if abs(fv - 1.0) < 0.11:
                fv = 1.0
            elif abs(fv - 0.5) < 0.11:
                fv = 0.5
            else:
                fv = 0.0
        pred_small[s] = fv
        out[s] = fv

    # cache only the non-zero predictions (or the full small dict)
    cache[key] = pred_small
    _save_cache(cache)
    return out


if __name__ == "__main__":
    sample = (
        "I designed and implemented infrastructure automation scripts using Terraform. "
        "I developed backend services with Express.js and optimized API response times."
    )
    pred = predict_zero_shot_baseline(sample, max_candidates=30)
    hits = [k for k, v in pred.items() if v > 0]
    print("Non-zero hits:", hits)
