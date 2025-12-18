from __future__ import annotations

import json
import random
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import time
from tqdm import tqdm
from openai import OpenAI

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from skills.globalVector import GLOBAL_SKILL_VECTOR
from skills.skillAliases import skills as SKILL_ALIASES
from skills.implicitCueBank import (
    get_category,
    skill_or_category_has_evidence,
    build_dynamic_implicit_sentence,
    style_hint,
    pick_required_evidence_phrases,
    text_contains_any_required_phrase,
)

# ---------------- CONFIG ----------------
OPENAI_MODEL = "gpt-4.1-nano"
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_FILE_BASE = "synthetic_dataset"

NUM_SAMPLES = 2
SEED = 2  # set None for non-deterministic runs

TEMPERATURE_GEN = 0.35
MAX_TOKENS_GEN = 280

TEMPERATURE_FIX = 0.0
MAX_TOKENS_FIX = 380

MAX_GEN_RETRIES = 2
MAX_FIX_RETRIES = 2

TARGET_IMPLICIT_RATIO = 0.50
BALANCE_HYSTERESIS = 0.06

K_MIN_JUNIOR = (3, 6)
K_MIN_MID = (4, 7)
K_MIN_SENIOR = (5, 9)

ROLE_FAMILIES = ["Software", "Data", "DevOps", "Security", "Product", "Management"]
SENIORITIES = ["Intern", "Junior", "Mid", "Senior", "Lead", "Manager"]
DOMAINS = ["FinTech", "E-commerce", "Healthcare", "Cyber", "SaaS", "Gaming"]
SENIORITY_WEIGHTS = [0.12, 0.22, 0.26, 0.20, 0.12, 0.08]

STATS = Counter()
TOTAL_CALLS = 0

NONSELECTED_REPLACEMENTS = {
    "AWS": "a major cloud provider",
    "Azure": "a major cloud provider",
    "Microservices": "service-based architecture",
    "Google Cloud": "a major cloud provider",
    "Kubernetes": "container orchestration",
    "Docker": "containerization",
    "SQL": "relational databases",
    "PostgreSQL": "relational databases",
    "MySQL": "relational databases",
    "MongoDB": "databases",
    "Redis": "caching",
    "Nginx": "a reverse proxy",
    "Grafana": "monitoring dashboards",
    "Prometheus": "monitoring metrics",
    "ELK Stack": "centralized logging",
}

# Strongly recommended: do not allow implicit for languages / markup / basic runtimes
DISALLOW_IMPLICIT_CATEGORIES = {
    "programming_language", "scripting_language", "markup_language", "backend_runtime"
}

# ---------------- GLOBAL VECTOR helper ----------------
def iter_skill_names() -> List[str]:
    if isinstance(GLOBAL_SKILL_VECTOR, dict):
        return list(GLOBAL_SKILL_VECTOR.keys())
    return list(GLOBAL_SKILL_VECTOR)

# ---------------- Versioned output ----------------
def get_next_versioned_output_file() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_versions = []
    for filename in os.listdir(OUTPUT_DIR):
        match = re.match(rf"^{re.escape(OUTPUT_FILE_BASE)}_v(\d+)\.jsonl$", filename)
        if match:
            existing_versions.append(int(match.group(1)))
    next_version = max(existing_versions) + 1 if existing_versions else 1
    return os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_BASE}_v{next_version}.jsonl")

# ---------------- Context ----------------
def sample_context(rng: random.Random) -> Dict[str, str]:
    seniority = rng.choices(SENIORITIES, weights=SENIORITY_WEIGHTS, k=1)[0]
    if seniority in ["Intern", "Junior"]:
        responsibility_depth = "IC"
    elif seniority == "Mid":
        responsibility_depth = rng.choice(["IC", "TechLead"])
    elif seniority in ["Senior", "Lead"]:
        responsibility_depth = rng.choice(["IC", "TechLead"])
    else:
        responsibility_depth = "PeopleManager"
    return {
        "role_family": rng.choice(ROLE_FAMILIES),
        "seniority": seniority,
        "responsibility_depth": responsibility_depth,
        "domain": rng.choice(DOMAINS),
    }

def load_prompt_templates() -> List[str]:
    fp = os.path.join(PROMPTS_DIR, "unified_prompt.txt")
    if not os.path.exists(fp):
        existing = sorted(os.listdir(PROMPTS_DIR)) if os.path.isdir(PROMPTS_DIR) else []
        raise RuntimeError(
            f"unified_prompt.txt not found in: {PROMPTS_DIR}\n"
            f"Existing files in Prompts: {existing}\n"
            f"Fix: make sure you have Prompts/unified_prompt.txt (exact name)."
        )
    with open(fp, "r", encoding="utf-8") as f:
        return [f.read()]

# ---------------- Skill surface matching ----------------
def _build_token_pattern(token: str) -> str:
    token = token.strip()
    esc = re.escape(token)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\.\+\-_/]*", token):
        return rf"\b{esc}\b"
    return rf"(?<!\w){esc}(?!\w)"

def _all_surface_forms(skill_name: str) -> List[str]:
    info = SKILL_ALIASES.get(skill_name, {})
    aliases = info.get("aliases", []) or []
    forms = [skill_name] + [a for a in aliases if a and a != skill_name]
    forms = sorted(set(forms), key=len, reverse=True)
    return forms

def count_skill_mentions(text_lower: str, skill_name: str) -> int:
    count = 0
    for form in _all_surface_forms(skill_name):
        pat = _build_token_pattern(form)
        count += len(re.findall(pat, text_lower, flags=re.IGNORECASE))
    return count

def find_nonselected_mentions(text_lower: str, selected_set: set) -> List[str]:
    hits = []
    for skill in iter_skill_names():
        if skill in selected_set:
            continue
        if count_skill_mentions(text_lower, skill) > 0:
            hits.append(skill)
    return hits

# ---------------- Balancer ----------------
class SkillBalanceController:
    def __init__(self, target: float = TARGET_IMPLICIT_RATIO, hysteresis: float = BALANCE_HYSTERESIS):
        self.target = target
        self.hysteresis = hysteresis
        self.explicit = 0
        self.implicit = 0

    def implicit_ratio(self) -> float:
        total = self.explicit + self.implicit
        return (self.implicit / total) if total else self.target

    def choose_n_implicit(self, n_total: int) -> int:
        if n_total <= 1:
            return 0
        base = int(round(n_total * self.target))
        base = max(0, min(n_total - 1, base))
        r = self.implicit_ratio()
        if r < self.target - self.hysteresis:
            base = min(n_total - 1, base + 1)
        elif r > self.target + self.hysteresis:
            base = max(0, base - 1)
        return base

    def update(self, selected_skills: Dict[str, float]) -> None:
        for _, label in selected_skills.items():
            if label == 0.5:
                self.implicit += 1
            elif label == 1.0:
                self.explicit += 1

# ---------------- Skill sampling ----------------
def _choose_k_by_seniority(rng: random.Random, seniority: str) -> int:
    if seniority in ["Intern", "Junior"]:
        lo, hi = K_MIN_JUNIOR
    elif seniority == "Mid":
        lo, hi = K_MIN_MID
    else:
        lo, hi = K_MIN_SENIOR
    return rng.randint(lo, hi)

def _skill_can_be_implicit(skill_name: str) -> bool:
    cat = get_category(SKILL_ALIASES, skill_name)
    if not cat:
        return False
    if cat in DISALLOW_IMPLICIT_CATEGORIES:
        return False
    # Hybrid: allow implicit only if skill or its category has evidence phrases
    return skill_or_category_has_evidence(skill_name, cat)

def pick_random_skills(
    rng: random.Random,
    controller: SkillBalanceController,
    ctx: Dict[str, str],
) -> Dict[str, float]:
    k = _choose_k_by_seniority(rng, ctx["seniority"])
    pool = iter_skill_names()
    chosen = rng.sample(pool, k)

    implicit_eligible = [s for s in chosen if _skill_can_be_implicit(s)]

    n_implicit = controller.choose_n_implicit(len(chosen))
    n_implicit = min(n_implicit, len(implicit_eligible))

    implicit_names = set(rng.sample(implicit_eligible, n_implicit)) if n_implicit else set()
    return {s: (0.5 if s in implicit_names else 1.0) for s in chosen}

# ---------------- Prompt building ----------------
def build_skills_info_and_requirements(
    rng: random.Random,
    selected_skills: Dict[str, float]
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """
    Returns:
    - skills_info JSON to inject into the prompt
    - implicit_requirements: skill -> required_evidence phrases
    """
    skills_info: Dict[str, Any] = {}
    implicit_requirements: Dict[str, List[str]] = {}

    for skill_name, label in selected_skills.items():
        aliases = (SKILL_ALIASES.get(skill_name, {}) or {}).get("aliases", [])[:8]
        banned_terms = [skill_name] + aliases

        if label == 1.0:
            skills_info[skill_name] = {"target_label": 1.0, "aliases": aliases}
            continue

        cat = get_category(SKILL_ALIASES, skill_name) or "unknown"

        # Generate varied "implicit_cues" sentences (NOT used for validation)
        cues = []
        for _ in range(rng.randint(2, 3)):
            cues.append(build_dynamic_implicit_sentence(rng, cat, skill_name))

        # REQUIRED evidence (this IS validated)
        required = pick_required_evidence_phrases(rng, skill_name, cat, 1, 2)
        implicit_requirements[skill_name] = required

        skills_info[skill_name] = {
            "target_label": 0.5,
            "category": cat,
            "implicit_cues": cues,
            "required_evidence": required,
            "banned_terms": banned_terms,
        }

    return skills_info, implicit_requirements

def generate_prompt(
    rng: random.Random,
    selected_skills: Dict[str, float],
    template: str,
    ctx: Dict[str, str],
    skills_info: Dict[str, Any],
) -> str:
    skills_block = json.dumps(skills_info, indent=2, ensure_ascii=False)
    base_prompt = template.replace("[PASTE_SKILLS_DICT_HERE]", skills_block)

    for k, v in ctx.items():
        base_prompt = base_prompt.replace("{" + k + "}", v)

    s_hint = style_hint(rng)

    return f"""
Return ONLY ONE resume-style paragraph (4–6 sentences), plain text.
No JSON. No code. No markdown. No headings. No meta language.

Hard rules:
- For target_label 1.0: MUST mention the canonical skill name OR an alias at least once.
- For target_label 0.5: MUST NOT mention the canonical name NOR any alias; MUST include at least one required_evidence phrase.
- Do NOT mention any skills beyond the provided list.

Style hint:
- {s_hint}

{base_prompt}
""".strip()

# ---------------- OpenAI call ----------------
def call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    global TOTAL_CALLS
    TOTAL_CALLS += 1

    client = OpenAI()
    backoff = 1.0
    for attempt in range(6):
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            return (resp.output_text or "").strip()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(12.0, backoff * 2)

# ---------------- Text utilities ----------------
def _count_sentences(text: str) -> int:
    parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(parts)

def _split_sentences_keep_punct(text: str) -> List[str]:
    sents = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text.strip())
    return [s.strip() for s in sents if s and s.strip()]

def trim_to_max_sentences(text: str, max_sentences: int = 6) -> str:
    sents = _split_sentences_keep_punct(text)
    if len(sents) <= max_sentences:
        return text.strip()
    trimmed = " ".join(sents[:max_sentences]).strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed

def cleanup_text(text: str) -> str:
    t = text
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+\.", ".", t)
    t = re.sub(r"\s+\!", "!", t)
    t = re.sub(r"\s+\?", "?", t)
    t = re.sub(r"\b(\w+)\s+\1\b", r"\1", t, flags=re.IGNORECASE)
    return t.strip()

def enforce_no_as_a_opening(text: str, rng: random.Random, allow_prob: float = 0.1) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if stripped.lower().startswith("as a") and rng.random() > allow_prob:
        text = re.sub(r"(?i)^\s*as a[^,.]*[,.]\s*", "", text).strip()
    return text

def post_process(text: str, rng: random.Random) -> str:
    text = enforce_no_as_a_opening(text, rng, allow_prob=0.1)
    text = cleanup_text(text)
    return text

# ---------------- Validation + Repairs ----------------
class ValidationError(Exception):
    def __init__(self, kind: str, skill: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.skill = skill

def validate_text(
    text: str,
    selected_skills: Dict[str, float],
    implicit_requirements: Dict[str, List[str]],
) -> str:
    if not text or len(text.strip()) < 40:
        raise ValidationError("text_invalid", "", "Text is empty/too short")

    text = text.strip()
    low = text.lower()

    banned_substrings = [
        "```", "import ", "def ", "class ", "print(", "here is", "output json", "```python",
        "based on the input", "corrected paragraph",
    ]
    if any(b in low for b in banned_substrings):
        raise ValidationError("text_invalid", "", "Meta/code detected in output")

    if any(ch in text for ch in ["{", "}", "[", "]"]):
        raise ValidationError("text_invalid", "", "Braces/brackets detected (likely JSON/code)")

    n_sent = _count_sentences(text)
    if n_sent > 6:
        text = trim_to_max_sentences(text, 6)
        low = text.lower()
        n_sent = _count_sentences(text)

    if n_sent < 4:
        raise ValidationError("text_invalid", "", f"Too few sentences: {n_sent} (expected 4–6)")

    # Selected skills checks
    for skill_name, label in selected_skills.items():
        mentions = count_skill_mentions(low, skill_name)

        if label == 1.0:
            if mentions == 0:
                raise ValidationError("explicit_missing", skill_name, f"Explicit skill missing: {skill_name}")
            if mentions > 4:
                raise ValidationError("explicit_too_many", skill_name, f"Explicit skill too many times: {skill_name} ({mentions})")

        else:
            # implicit must not be named
            if mentions > 0:
                raise ValidationError("implicit_leaked", skill_name, f"Implicit skill leaked: {skill_name}")

            required = implicit_requirements.get(skill_name, [])
            if required and not text_contains_any_required_phrase(text, required):
                raise ValidationError("implicit_no_evidence", skill_name, f"No implicit evidence for: {skill_name}")

    # No extra skills
    selected_set = set(selected_skills.keys())
    nonselected_hits = find_nonselected_mentions(low, selected_set)
    if nonselected_hits:
        raise ValidationError("nonselected_mentioned", nonselected_hits[0], f"Mentions non-selected skill: {nonselected_hits[0]}")

    return text

def sanitize_skill_mention(text: str, skill: str) -> str:
    replacement = NONSELECTED_REPLACEMENTS.get(skill, "standard engineering practices")
    candidates = _all_surface_forms(skill)
    out = text
    for cand in candidates:
        pat = r"(?i)(?<!\w)" + re.escape(cand) + r"(?!\w)"
        out = re.sub(pat, replacement, out)
    return cleanup_text(out)

def build_fix_prompt(original_text: str, skills_info: Dict[str, Any]) -> str:
    skills_block = json.dumps(skills_info, indent=2, ensure_ascii=False)
    return f"""
Return ONLY the corrected paragraph (plain text), 4–6 sentences.
No JSON. No code. No markdown. No headings. No meta text.

Rules:
- target_label 1.0: MUST mention canonical name OR one alias at least ONCE.
- target_label 0.5: MUST NOT mention canonical name NOR any alias from banned_terms.
- target_label 0.5: MUST include at least ONE phrase from required_evidence exactly as written (case-insensitive OK).
- Do NOT add any skills beyond the list.

skills_info:
{skills_block}

Original paragraph:
{original_text}
""".strip()

def force_add_missing_explicit(text: str, missing_skill: str) -> str:
    sents = _split_sentences_keep_punct(text)
    injection = f"I used {missing_skill} in production work and ongoing maintenance."
    if len(sents) < 2:
        return cleanup_text(text + " " + injection)
    insert_at = max(1, len(sents) - 1)
    new_sents = sents[:insert_at] + [injection] + sents[insert_at:]
    return cleanup_text(" ".join(new_sents))

# ---------------- Generation ----------------
def _choose_k_by_seniority(rng: random.Random, seniority: str) -> int:
    if seniority in ["Intern", "Junior"]:
        lo, hi = K_MIN_JUNIOR
    elif seniority == "Mid":
        lo, hi = K_MIN_MID
    else:
        lo, hi = K_MIN_SENIOR
    return rng.randint(lo, hi)

def generate_sample(
    rng: random.Random,
    templates: List[str],
    controller: SkillBalanceController
) -> Optional[Dict[str, Any]]:
    ctx = sample_context(rng)
    selected_skills = pick_random_skills(rng, controller, ctx)
    template = rng.choice(templates)

    skills_info, implicit_requirements = build_skills_info_and_requirements(rng, selected_skills)
    prompt = generate_prompt(rng, selected_skills, template, ctx, skills_info)

    last_err: Optional[Exception] = None
    text = ""

    # 1) initial generation attempts
    for _ in range(MAX_GEN_RETRIES):
        try:
            text = call_openai(prompt, TEMPERATURE_GEN, MAX_TOKENS_GEN)
            text = post_process(text, rng)
            text = validate_text(text, selected_skills, implicit_requirements)
            STATS["pass_initial"] += 1
            controller.update(selected_skills)
            return {"job_description": text, "skills": selected_skills}
        except ValidationError as ve:
            last_err = ve
        except Exception as e:
            last_err = e

    # 2) cheap local repair loop
    repaired_text = text
    for _ in range(12):
        try:
            repaired_text = post_process(repaired_text, rng)
            repaired_text = validate_text(repaired_text, selected_skills, implicit_requirements)
            STATS["pass_repair"] += 1
            controller.update(selected_skills)
            return {"job_description": repaired_text, "skills": selected_skills}

        except ValidationError as ve:
            last_err = ve

            if ve.kind == "explicit_missing" and ve.skill:
                STATS["repair_add_explicit"] += 1
                repaired_text = force_add_missing_explicit(repaired_text, ve.skill)
                continue

            if ve.kind == "explicit_too_many" and ve.skill:
                STATS["repair_soften_repeats"] += 1
                repaired_text = sanitize_skill_mention(repaired_text, ve.skill)
                repaired_text = force_add_missing_explicit(repaired_text, ve.skill)
                continue

            if ve.kind in {"implicit_leaked", "nonselected_mentioned"} and ve.skill:
                STATS["repair_sanitize"] += 1
                repaired_text = sanitize_skill_mention(repaired_text, ve.skill)
                continue

            
            if ve.kind == "implicit_no_evidence" and ve.skill:
                # Drop the implicit skill (no hallucinated 0.5). Do NOT validate+return here,
                # because the text may still contain other violations (e.g., non-selected skills).
                STATS["repair_drop_implicit_skill"] += 1
                selected_skills.pop(ve.skill, None)
                implicit_requirements.pop(ve.skill, None)
                continue


            break

        except Exception as e:
            last_err = e
            break

    # 3) LLM fix pass
    fix_prompt = build_fix_prompt(repaired_text, skills_info)
    fixed_text = repaired_text

    for _ in range(MAX_FIX_RETRIES):
        try:
            fixed_text = call_openai(fix_prompt, TEMPERATURE_FIX, MAX_TOKENS_FIX)
            fixed_text = post_process(fixed_text, rng)
            fixed_text = validate_text(fixed_text, selected_skills, implicit_requirements)
            STATS["pass_fix_llm"] += 1
            controller.update(selected_skills)
            return {"job_description": fixed_text, "skills": selected_skills}

        except ValidationError as ve:
            last_err = ve

            if ve.kind == "explicit_missing" and ve.skill:
                STATS["fix_add_explicit"] += 1
                fixed_text = force_add_missing_explicit(fixed_text, ve.skill)
                fix_prompt = build_fix_prompt(fixed_text, skills_info)
                continue

            if ve.kind in {"implicit_leaked", "nonselected_mentioned"} and ve.skill:
                STATS["fix_sanitize"] += 1
                fixed_text = sanitize_skill_mention(fixed_text, ve.skill)
                fix_prompt = build_fix_prompt(fixed_text, skills_info)
                continue

            if ve.kind == "implicit_no_evidence" and ve.skill:
                STATS["fix_drop_implicit_skill"] += 1
                selected_skills.pop(ve.skill, None)
                implicit_requirements.pop(ve.skill, None)
                fixed_text = validate_text(fixed_text, selected_skills, implicit_requirements)
                controller.update(selected_skills)
                return {"job_description": fixed_text, "skills": selected_skills}

        except Exception as e:
            last_err = e
            break

    STATS["skip"] += 1
    print(f"[SKIP] Failed: {str(last_err) if last_err else 'unknown'}")
    return None

def pick_random_skills(
    rng: random.Random,
    controller: SkillBalanceController,
    ctx: Dict[str, str],
) -> Dict[str, float]:
    k = _choose_k_by_seniority(rng, ctx["seniority"])
    pool = iter_skill_names()
    chosen = rng.sample(pool, k)

    implicit_eligible = [s for s in chosen if _skill_can_be_implicit(s)]
    n_implicit = controller.choose_n_implicit(len(chosen))
    n_implicit = min(n_implicit, len(implicit_eligible))

    implicit_names = set(rng.sample(implicit_eligible, n_implicit)) if n_implicit else set()
    return {s: (0.5 if s in implicit_names else 1.0) for s in chosen}

def main():
    rng = random.Random(SEED) if SEED is not None else random.Random()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    templates = load_prompt_templates()
    controller = SkillBalanceController()
    output_file = get_next_versioned_output_file()

    print(f"Model: {OPENAI_MODEL}")
    print(f"Prompts dir: {PROMPTS_DIR}")
    print(f"Output: {output_file}")
    print(f"Loaded {len(templates)} templates")
    print(f"Target implicit ratio: {TARGET_IMPLICIT_RATIO:.0%}")

    generated = 0
    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_SAMPLES)
        while generated < NUM_SAMPLES:
            sample = generate_sample(rng, templates, controller)
            if sample:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
                generated += 1
                pbar.update(1)
        pbar.close()

    print(f"\nDone. Wrote {generated} samples to {output_file}")
    total_labels = controller.explicit + controller.implicit
    if total_labels:
        print(f"Implicit labels: {controller.implicit} | Explicit labels: {controller.explicit} | Implicit ratio: {controller.implicit/total_labels:.2%}")
    else:
        print("Implicit labels: 0 | Explicit labels: 0 | Implicit ratio: N/A")
    print(f"Total model calls: {TOTAL_CALLS}")
    for k, v in STATS.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
