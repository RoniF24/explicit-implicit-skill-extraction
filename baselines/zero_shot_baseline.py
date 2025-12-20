# baselines/zero_shot_baseline.py
from __future__ import annotations

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
import urllib.request
import urllib.error

# Make project root importable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR

# ---------------- Backend config ----------------
# Choose backend: "ollama" (default) or "openai"
ZERO_SHOT_BACKEND = os.getenv("ZERO_SHOT_BACKEND", "ollama").strip().lower()

# ---------------- Ollama config ----------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TEMPERATURE = float(os.getenv("ZERO_SHOT_TEMPERATURE", os.getenv("OPENAI_TEMPERATURE", "0")))

# ---------------- OpenAI config (optional) ----------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

# Cache
DEFAULT_CACHE_DIR = "outputs/cache/zero_shot_ollama" if ZERO_SHOT_BACKEND == "ollama" else "outputs/cache/zero_shot_openai"
CACHE_DIR = Path(os.getenv("ZERO_SHOT_CACHE_DIR", DEFAULT_CACHE_DIR))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TAG = OLLAMA_MODEL if ZERO_SHOT_BACKEND == "ollama" else OPENAI_MODEL
SAFE_TAG = MODEL_TAG.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
CACHE_PATH = CACHE_DIR / f"zero_shot_cache__{ZERO_SHOT_BACKEND}__{SAFE_TAG}.json"

# Global candidates
GLOBAL_CANDIDATES: List[str] = [s for s in GLOBAL_SKILL_VECTOR if isinstance(s, str) and s.strip()]
GLOBAL_SET = set(GLOBAL_CANDIDATES)

# Prompt template file
PROMPT_PATH = Path(ROOT_DIR) / "Prompts" / "zero_shot_system_prompt.txt"


def _hash_key(text: str) -> str:
    # Include backend+model in hash to avoid cross-backend collisions
    payload = f"{ZERO_SHOT_BACKEND}::{MODEL_TAG}::{text.strip()}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


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


def _build_system_prompt(global_candidates: List[str]) -> str:
    cand_str = ", ".join(global_candidates)
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.format(cand_str=cand_str).strip()


SYSTEM_PROMPT = _build_system_prompt(GLOBAL_CANDIDATES)


def _ollama_generate(idx: int, resume_text: str) -> str:
    """
    Calls Ollama local server.
    Uses /api/chat with system+user messages.
    """
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'idx: {idx}\n\nRESUME CHUNK:\n"""{resume_text}"""'},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE},
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            obj = json.loads(raw)
            # Ollama chat response schema: {"message":{"role":"assistant","content":"..."}, ...}
            msg = obj.get("message", {}) if isinstance(obj, dict) else {}
            return str(msg.get("content", "") or "")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama connection failed. Is Ollama running on {OLLAMA_HOST}? Error: {e}"
        )


def _openai_generate(idx: int, resume_text: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI SDK not installed/available, but ZERO_SHOT_BACKEND=openai") from e

    client = OpenAI()  # reads OPENAI_API_KEY from env

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'idx: {idx}\n\nRESUME CHUNK:\n"""{resume_text}"""'},
        ],
        temperature=TEMPERATURE,
    )

    # best-effort extraction
    if getattr(resp, "output_text", None):
        return resp.output_text or ""

    try:
        return resp.output[0].content[0].text or ""
    except Exception:
        return ""


def _llm_generate(idx: int, resume_text: str) -> str:
    if ZERO_SHOT_BACKEND == "ollama":
        return _ollama_generate(idx, resume_text)
    if ZERO_SHOT_BACKEND == "openai":
        return _openai_generate(idx, resume_text)
    raise ValueError("ZERO_SHOT_BACKEND must be: ollama or openai")


def predict_from_text(idx: int, job_description: str) -> Dict[str, Any]:
    """
    Input: resume chunk text only
    Output:
      {
        "idx": idx,
        "skills": { "Skill": 1.0 or 0.5, ... }   # omitted = 0
      }
    """
    text = (job_description or "").strip()
    out = {"idx": idx, "skills": {}}

    if not text:
        return out

    cache = _load_cache()
    key = _hash_key(text)

    if key in cache and isinstance(cache[key], dict):
        cached = cache[key]
        skills = cached.get("skills", {})
        out["skills"] = skills if isinstance(skills, dict) else {}
        return out

    raw = _llm_generate(idx, text)
    parsed = _extract_json_object(raw)

    if not parsed or not isinstance(parsed, dict):
        cache[key] = out
        _save_cache(cache)
        return out

    skills = parsed.get("skills", {})
    if not isinstance(skills, dict):
        skills = {}

    cleaned: Dict[str, float] = {}

    # enforce closed list + allowed labels
    for k, v in skills.items():
        if not isinstance(k, str) or k not in GLOBAL_SET:
            continue
        if v not in (1, 1.0, 0.5):
            continue
        cleaned[k] = float(v)

    # enforce: if skill name appears in text => must be explicit (1.0)
    low_text = text.lower()
    for skill, label in list(cleaned.items()):
        if label == 0.5 and skill.lower() in low_text:
            cleaned[skill] = 1.0

    out["skills"] = cleaned

    cache[key] = out
    _save_cache(cache)
    return out


def predict_zero_shot_baseline(text: str) -> Dict[str, float]:
    """
    Input: resume/job description text
    Output: dict of skills {skill: 1.0 or 0.5} (only non-zero), closed list enforced inside.
    """
    return predict_from_text(idx=0, job_description=text).get("skills", {})


def _read_jsonl(path: str) -> List[tuple[int, dict]]:
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
    ap.add_argument("--in_path", required=True, help="Input JSONL dataset (each line has job_description/resume_chunk_text/text)")
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
            pred = predict_from_text(idx=idx, job_description=str(text))
            wf.write(json.dumps(pred, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote predictions: {n} -> {outp.resolve()}")
    print(f"Backend: {ZERO_SHOT_BACKEND} | Model: {MODEL_TAG}")
    print(f"Cache file: {CACHE_PATH.resolve()}")


if __name__ == "__main__":
    main()
