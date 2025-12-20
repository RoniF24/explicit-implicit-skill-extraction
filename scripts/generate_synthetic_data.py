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
from skills.implicitCueBank import skill_has_evidence  # להוסיף ל-imports
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
OPENAI_MODEL = "gpt-4o-mini"
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_FILE_BASE = "synthetic_dataset"

NUM_SAMPLES = 20
SEED = 5  # set None for non-deterministic runs

TEMPERATURE_GEN = 0.35
MAX_TOKENS_GEN = 280

TEMPERATURE_FIX = 0.0
MAX_TOKENS_FIX = 380

MAX_GEN_RETRIES = 2
MAX_FIX_RETRIES = 2

TARGET_IMPLICIT_RATIO = 0.50
BALANCE_HYSTERESIS = 0.06

# Hard cap to keep implicit (0.5) evidence clean in a 4–6 sentence paragraph.
MAX_IMPLICIT_PER_SAMPLE = 2

K_MIN_JUNIOR = (3, 6)
K_MIN_MID = (4, 7)
K_MIN_SENIOR = (5, 9)

ROLE_FAMILIES = ["Software", "Data", "DevOps", "Security", "Product", "ML"]
DOMAINS = ["FinTech", "E-commerce", "Healthcare", "Cyber", "SaaS", "Gaming"]
SENIORITY_LEVELS = ["Intern", "Junior", "Mid", "Senior", "Staff", "Lead"]
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
    "JavaScript": "a scripting language",
}

CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---------------- IO helpers ----------------
def load_prompt_template() -> str:
    prompt_path = os.path.join(PROMPTS_DIR, "unified_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_templates() -> List[str]:
    tpl_path = os.path.join(PROMPTS_DIR, "templates.txt")
    if not os.path.exists(tpl_path):
        return [
            "Write a resume-style paragraph describing work experience in {domain}.",
            "Write a resume-style paragraph for a {role_family} engineer in {domain}.",
        ]
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpls = [ln.strip() for ln in f.readlines() if ln.strip()]
    return tpls or [
        "Write a resume-style paragraph describing work experience in {domain}.",
        "Write a resume-style paragraph for a {role_family} engineer in {domain}.",
    ]

def get_next_versioned_output_file() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing = [fn for fn in os.listdir(OUTPUT_DIR) if fn.startswith(OUTPUT_FILE_BASE) and fn.endswith(".jsonl")]
    versions = []
    for fn in existing:
        m = re.search(r"_v(\d+)\.jsonl$", fn)
        if m:
            versions.append(int(m.group(1)))
    next_version = (max(versions) + 1) if versions else 1
    return os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_BASE}_v{next_version}.jsonl")

# ---------------- Skill listing ----------------
GLOBAL_SKILL_SET = set(GLOBAL_SKILL_VECTOR)

def iter_skill_names() -> List[str]:
    return list(GLOBAL_SKILL_VECTOR)

def _all_surface_forms(skill_name: str) -> List[str]:
    forms = [skill_name]
    aliases = (SKILL_ALIASES.get(skill_name, {}) or {}).get("aliases", [])
    for a in aliases:
        if a and a not in forms:
            forms.append(a)
    return forms[:10]

# ---------------- Text helpers ----------------
def cleanup_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text

def _split_sentences_keep_punct(text: str) -> List[str]:
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

def enforce_no_as_a_opening(text: str, rng: random.Random, allow_prob: float = 0.1) -> str:
    low = text.strip().lower()
    if low.startswith("as a "):
        if rng.random() > allow_prob:
            # rewrite prefix lightly
            text = re.sub(r"(?i)^as a\s+", "", text.strip())
            text = text[0].upper() + text[1:] if text else text
    return text

def post_process(text: str, rng: random.Random) -> str:
    text = enforce_no_as_a_opening(text, rng, allow_prob=0.1)
    text = cleanup_text(text)
    return text

# ---------------- Matching ----------------
def count_skill_mentions(text_lower: str, skill_name: str) -> int:
    candidates = _all_surface_forms(skill_name)
    count = 0
    for cand in candidates:
        pat = r"(?<!\w)" + re.escape(cand.lower()) + r"(?!\w)"
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
    def __init__(self, target: float = TARGET_IMPLICIT_RATIO, hysteresis: float = BALANCE_HYSTERESIS, max_implicit_per_sample: int = MAX_IMPLICIT_PER_SAMPLE):
        self.target = target
        self.hysteresis = hysteresis
        self.max_implicit_per_sample = max_implicit_per_sample
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
        base = min(base, self.max_implicit_per_sample)
        return base

    def update(self, selected_skills: Dict[str, float]) -> None:
        for _, lab in selected_skills.items():
            if lab == 1.0:
                self.explicit += 1
            elif lab == 0.5:
                self.implicit += 1

# ---------------- Sampling context ----------------
def sample_context(rng: random.Random) -> Dict[str, str]:
    role_family = rng.choice(ROLE_FAMILIES)
    domain = rng.choice(DOMAINS)
    seniority = rng.choices(SENIORITY_LEVELS, weights=SENIORITY_WEIGHTS, k=1)[0]
    return {"role_family": role_family, "domain": domain, "seniority": seniority}

# ---------------- OpenAI call ----------------
def call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    global TOTAL_CALLS
    TOTAL_CALLS += 1
    resp = CLIENT.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    # Responses API: text is under output[0].content[0].text typically
    try:
        return resp.output[0].content[0].text
    except Exception:
        # fallback
        return str(resp)

# ---------------- Repairs helpers ----------------
def force_add_missing_explicit(text: str, skill_name: str) -> str:
    """Ensure explicit skill appears at least once (canonical name)."""
    if not text:
        return skill_name
    sents = _split_sentences_keep_punct(text)
    injection = f"I used {skill_name} in production work and ongoing maintenance."
    if any(skill_name.lower() in s.lower() for s in sents):
        return text
    insert_at = max(0, len(sents) - 1)
    new_sents = sents[:insert_at] + [injection] + sents[insert_at:]
    return cleanup_text(" ".join(new_sents))

def soften_repeated_explicit(text: str, skill_name: str, max_mentions: int = 3) -> str:
    """If explicit is repeated too many times, replace extra mentions with generic phrasing."""
    low = text.lower()
    count = count_skill_mentions(low, skill_name)
    if count <= max_mentions:
        return text
    replacement = "standard engineering practices"
    # Replace all but first mention
    candidates = _all_surface_forms(skill_name)
    out = text
    seen = 0
    for cand in candidates:
        pat = r"(?i)(?<!\w)" + re.escape(cand) + r"(?!\w)"
        def _repl(m):
            nonlocal seen
            seen += 1
            return cand if seen <= 1 else replacement
        out = re.sub(pat, _repl, out)
    return cleanup_text(out)

def sanitize_skill_mention(text: str, skill: str) -> str:
    replacement = NONSELECTED_REPLACEMENTS.get(skill, "standard engineering practices")
    candidates = _all_surface_forms(skill)
    out = text
    for cand in candidates:
        pat = r"(?i)(?<!\w)" + re.escape(cand) + r"(?!\w)"
        out = re.sub(pat, replacement, out)
    return cleanup_text(out)

def force_add_missing_implicit_evidence(text: str, skill_name: str, required_phrases: List[str]) -> str:
    """
    Inject a short sentence that includes ONE required evidence phrase EXACTLY as written.
    - Does NOT mention the skill name (or aliases).
    - Keeps resume tone.
    """
    if not required_phrases:
        return text

    phrase = required_phrases[0]  # stable for debugging

    injection = f"I applied {phrase} to improve reliability under tight SLAs."

    sents = _split_sentences_keep_punct(text)
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
            STATS["pass_gen"] += 1
            controller.update(selected_skills)
            return {"job_description": text, "skills": selected_skills}
        except ValidationError as ve:
            last_err = ve
            STATS[f"gen_fail_{ve.kind}"] += 1
        except Exception as e:
            last_err = e
            break

    # 2) local repair pass (no LLM)
    repaired_text = text if text else ""
    for _ in range(3):
        try:
            repaired_text = validate_text(repaired_text, selected_skills, implicit_requirements)
            STATS["pass_repair_local"] += 1
            controller.update(selected_skills)
            return {"job_description": repaired_text, "skills": selected_skills}

        except ValidationError as ve:
            last_err = ve

            if ve.kind == "text_invalid":
                break

            if ve.kind == "explicit_missing" and ve.skill:
                STATS["repair_add_explicit"] += 1
                repaired_text = force_add_missing_explicit(repaired_text, ve.skill)
                continue

            if ve.kind == "explicit_too_many" and ve.skill:
                STATS["repair_soften_repeats"] += 1
                repaired_text = soften_repeated_explicit(repaired_text, ve.skill)
                continue

            if ve.kind in {"implicit_leaked", "nonselected_mentioned"} and ve.skill:
                STATS["repair_sanitize"] += 1
                repaired_text = sanitize_skill_mention(repaired_text, ve.skill)
                continue

            if ve.kind == "implicit_no_evidence" and ve.skill:
                # Prefer injecting required evidence to keep the 0.5 label stable.
                req = implicit_requirements.get(ve.skill, [])
                if req:
                    STATS["repair_add_missing_implicit_evidence"] += 1
                    repaired_text = force_add_missing_implicit_evidence(repaired_text, ve.skill, req)
                    continue

                # Fallback: if we truly have no evidence phrases, drop the implicit skill
                STATS["repair_drop_implicit_skill"] += 1
                selected_skills.pop(ve.skill, None)
                implicit_requirements.pop(ve.skill, None)
                skills_info.pop(ve.skill, None)
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
                # Validate locally first; avoid another LLM call if we already pass.
                try:
                    fixed_text = validate_text(fixed_text, selected_skills, implicit_requirements)
                    STATS["pass_fix_local_inject"] += 1
                    controller.update(selected_skills)
                    return {"job_description": fixed_text, "skills": selected_skills}
                except ValidationError:
                    fix_prompt = build_fix_prompt(fixed_text, skills_info)
                    continue

            if ve.kind in {"implicit_leaked", "nonselected_mentioned"} and ve.skill:
                STATS["fix_sanitize"] += 1
                fixed_text = sanitize_skill_mention(fixed_text, ve.skill)
                try:
                    fixed_text = validate_text(fixed_text, selected_skills, implicit_requirements)
                    STATS["pass_fix_local_sanitize"] += 1
                    controller.update(selected_skills)
                    return {"job_description": fixed_text, "skills": selected_skills}
                except ValidationError:
                    fix_prompt = build_fix_prompt(fixed_text, skills_info)
                    continue

            if ve.kind == "implicit_no_evidence" and ve.skill:
                req = implicit_requirements.get(ve.skill, [])
                if req:
                    STATS["fix_add_missing_implicit_evidence"] += 1
                    fixed_text = force_add_missing_implicit_evidence(fixed_text, ve.skill, req)
                else:
                    # No evidence phrases available -> drop the implicit skill safely.
                    STATS["fix_drop_implicit_skill"] += 1
                    selected_skills.pop(ve.skill, None)
                    implicit_requirements.pop(ve.skill, None)
                    skills_info.pop(ve.skill, None)

                try:
                    fixed_text = validate_text(fixed_text, selected_skills, implicit_requirements)
                    STATS["pass_fix_local_implicit"] += 1
                    controller.update(selected_skills)
                    return {"job_description": fixed_text, "skills": selected_skills}
                except ValidationError:
                    fix_prompt = build_fix_prompt(fixed_text, skills_info)
                    continue

            # Unknown / unrecoverable validation error in fix loop
            break

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

    out: Dict[str, float] = {}
    for s in chosen:
        out[s] = 0.5 if s in implicit_names else 1.0
    return out

def _skill_can_be_implicit(skill_name: str) -> bool:
    category = get_category(SKILL_ALIASES, skill_name)
    return skill_or_category_has_evidence(skill_name, category)

def build_skills_info_and_requirements(
    rng: random.Random,
    selected_skills: Dict[str, float]
) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
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
        required = pick_required_evidence_phrases(rng, skill_name, cat, k_min=1, k_max=2)
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
    skills_info: Dict[str, Any]
) -> str:
    base = load_prompt_template()
    rendered_tpl = template.format(**ctx)

    # Put a short style hint to increase variety
    hint = style_hint(rng)

    prompt = base.replace("[TEMPLATE]", rendered_tpl).replace("[STYLE_HINT]", hint)
    prompt = prompt.replace("[PASTE_SKILLS_DICT_HERE]", json.dumps(skills_info, ensure_ascii=False, indent=2))
    return prompt

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
        "{", "}", "[", "]",
    ]
    for b in banned_substrings:
        if b in low:
            raise ValidationError("banned_content", "", f"Banned content: {b}")

    # enforce 4–6 sentences
    sents = _split_sentences_keep_punct(text)
    if len(sents) < 4 or len(sents) > 6:
        raise ValidationError("sentence_count", "", f"Bad sentence count: {len(sents)}")

    selected_set = set(selected_skills.keys())

    # per-skill checks
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

    # non-selected should not be mentioned
    nonselected_hits = find_nonselected_mentions(low, selected_set)
    if nonselected_hits:
        raise ValidationError("nonselected_mentioned", nonselected_hits[0], f"Mentions non-selected skill: {nonselected_hits[0]}")

    return text

def build_fix_prompt(original_text: str, skills_info: Dict[str, Any]) -> str:
    """
    Give the LLM the original text + skill spec, and ask it to minimally fix violations.
    """
    base = load_prompt_template()

    instructions = (
        "Fix the paragraph to satisfy ALL constraints. "
        "Keep the meaning and resume tone. "
        "Do not add any new skills/tools not in the skills spec. "
        "If an implicit skill is required, include at least one required_evidence phrase EXACTLY as written, "
        "but do not mention the skill name/aliases."
    )

    fix_prompt = (
        base
        + "\n\n[FIX_INSTRUCTIONS]\n"
        + instructions
        + "\n\n[ORIGINAL_PARAGRAPH]\n"
        + original_text
        + "\n\n[PASTE_SKILLS_DICT_HERE]\n"
        + json.dumps(skills_info, ensure_ascii=False, indent=2)
    )
    return fix_prompt

# ---------------- Main ----------------
def main() -> None:
    rng = random.Random(SEED) if SEED is not None else random.Random()
    templates = load_templates()
    controller = SkillBalanceController()

    out_path = get_next_versioned_output_file()
    written = 0
    attempts = 0

    # Safety cap כדי לא להיתקע לנצח אם משהו ממש לא יציב (אפשר לשנות/להסיר)
    max_attempts = NUM_SAMPLES * 20  # למשל עד 1020 ניסיונות בשביל 51 דוגמאות

    pbar = tqdm(total=NUM_SAMPLES)

    with open(out_path, "w", encoding="utf-8") as f:
        while written < NUM_SAMPLES and attempts < max_attempts:
            attempts += 1
            sample = generate_sample(rng, templates, controller)
            if not sample:
                continue

            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1
            pbar.update(1)

    pbar.close()

    print(f"\nWrote {written} samples to: {out_path}")
    print(f"Attempts: {attempts}")
    print(f"Total model calls: {TOTAL_CALLS}")
    print("Stats:")
    for k, v in STATS.most_common():
        print(f"  {k}: {v}")

    if written < NUM_SAMPLES:
        print(f"\n[WARN] Stopped early: reached max_attempts={max_attempts}. "
              f"Increase max_attempts or reduce skips.")


if __name__ == "__main__":
    main()
