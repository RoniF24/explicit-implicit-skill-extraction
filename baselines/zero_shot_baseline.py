# baselines/zero_shot_baseline.py
from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from openai import OpenAI

# ---------------- OpenAI config ----------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

# Cache (so reruns don't re-pay / re-wait)
CACHE_DIR = Path(os.getenv("ZERO_SHOT_CACHE_DIR", "outputs/cache/zero_shot_openai"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / f"zero_shot_cache__{OPENAI_MODEL.replace(':','_').replace('/','_')}.json"

client = OpenAI()


def _hash_key(text: str, candidates: List[str]) -> str:
    blob = text.strip() + "||" + "||".join(candidates)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


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
    """Try to find and parse the first JSON object in model output."""
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


def _build_prompt(idx: int, text: str, candidates: List[str]) -> str:
    # Candidates are ONLY the skills from this row (GT keys), without labels.
    # Output must NOT invent new skills, and must NOT output 0.
    cand_str = ", ".join(candidates)

    return f"""
You are labeling skills for ONE example.

You MUST choose only from the given candidate skills list (closed list).
Do NOT add new skills.

Return ONLY skills that have evidence:
- EXPLICIT (1.0): the skill name appears clearly in the text.
- IMPLICIT (0.5): the skill is strongly implied by responsibilities/processes, but NOT mentioned by name.

IMPORTANT:
- Do NOT output 0.0 skills at all (omit them).
- If the skill name appears in the text, it MUST be EXPLICIT (1.0), not IMPLICIT.
- Output JSON ONLY. No extra text.

Example id (idx): {idx}

CANDIDATE SKILLS (closed list):
[{cand_str}]

JOB DESCRIPTION:
\"\"\"{text}\"\"\"

Return exactly this JSON schema:
{{
  "idx": {idx},
  "explicit": ["skill1", "skill2"],
  "implicit": ["skill3", "skill4"]
}}
""".strip()


def _openai_generate(prompt: str) -> str:
    # Responses API
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": "You are a careful JSON-only classifier. Follow the schema exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
    )
    # SDK exposes a convenience text accessor
    return getattr(resp, "output_text", "") or ""


def predict_from_row(idx: int, job_description: str, candidate_skills: List[str]) -> Dict[str, Any]:
    """
    Returns:
      {
        "idx": idx,
        "explicit": [...],   # each implies label 1.0
        "implicit": [...],   # each implies label 0.5
      }
    """
    text = (job_description or "").strip()
    candidates = [c.strip() for c in candidate_skills if isinstance(c, str) and c.strip()]
    # de-dup but keep stable order
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    out = {"idx": idx, "explicit": [], "implicit": []}

    if not text or not candidates:
        return out

    cache = _load_cache()
    key = _hash_key(text, candidates)

    if key in cache and isinstance(cache[key], dict):
        cached = cache[key]
        # ensure schema exists
        out["explicit"] = list(cached.get("explicit", [])) if isinstance(cached.get("explicit", []), list) else []
        out["implicit"] = list(cached.get("implicit", [])) if isinstance(cached.get("implicit", []), list) else []
        return out

    prompt = _build_prompt(idx, text, candidates)
    raw = _openai_generate(prompt)
    parsed = _extract_json_object(raw)

    if not parsed or not isinstance(parsed, dict):
        cache[key] = out
        _save_cache(cache)
        return out

    explicit = parsed.get("explicit", [])
    implicit = parsed.get("implicit", [])

    if not isinstance(explicit, list):
        explicit = []
    if not isinstance(implicit, list):
        implicit = []

    # enforce closed list + no overlaps
    cand_set = set(candidates)
    explicit_clean = [s for s in explicit if isinstance(s, str) and s in cand_set]
    implicit_clean = [s for s in implicit if isinstance(s, str) and s in cand_set]

    exp_set = set(explicit_clean)
    implicit_clean = [s for s in implicit_clean if s not in exp_set]

    out["explicit"] = explicit_clean
    out["implicit"] = implicit_clean

    cache[key] = out
    _save_cache(cache)
    return out


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append((i, obj))
    return rows


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input JSONL dataset (each line has job_description + skills dict)")
    ap.add_argument("--out_path", required=True, help="Where to write predictions JSONL")
    ap.add_argument("--limit", type=int, default=0, help="Run only first N rows (0 = all)")
    args = ap.parse_args()

    inp = args.in_path
    outp = Path(args.out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(inp)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    n = 0
    with outp.open("w", encoding="utf-8") as wf:
        for idx, obj in rows:
            text = obj.get("job_description") or obj.get("resume_chunk_text") or obj.get("text") or ""
            skills_dict = obj.get("skills", {}) if isinstance(obj.get("skills", {}), dict) else {}
            candidates = list(skills_dict.keys())

            pred = predict_from_row(idx=idx, job_description=str(text), candidate_skills=candidates)
            wf.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote predictions: {n} -> {outp.resolve()}")
    print(f"Cache file: {CACHE_PATH.resolve()}")


if __name__ == "__main__":
    main()
