import json
import random
import os
import sys
from typing import List, Dict, Any, Optional
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

# ---------------- CONFIG ----------------
OPENAI_MODEL = "gpt-4.1-nano"  # or "gpt-4.1-nano-2025-04-14" for snapshot stability
PROMPTS_DIR = os.path.join(ROOT_DIR, "Prompts")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_FILE_BASE = "synthetic_dataset"  # Base name without version

NUM_SAMPLES = 80


def get_next_versioned_output_file() -> str:
    """
    Find the next available version for synthetic_dataset.
    Returns path like: data/synthetic_dataset_v1.jsonl, data/synthetic_dataset_v2.jsonl, etc.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find existing versioned files
    existing_versions = []
    for filename in os.listdir(OUTPUT_DIR):
        # Match pattern: synthetic_dataset_v{N}.jsonl
        match = re.match(rf"^{re.escape(OUTPUT_FILE_BASE)}_v(\d+)\.jsonl$", filename)
        if match:
            existing_versions.append(int(match.group(1)))
    
    # Determine next version
    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        next_version = 1
    
    return os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_BASE}_v{next_version}.jsonl")

# generation params
TEMPERATURE_GEN = 0.2
MAX_TOKENS_GEN = 220  # give a bit more headroom for 4–6 sentences

# fix pass params (more deterministic)
TEMPERATURE_FIX = 0.0
MAX_TOKENS_FIX = 260

# retries
MAX_GEN_RETRIES = 3
MAX_FIX_RETRIES = 3

# Prefer explicit to reduce leakage + nonselected implied failures
P_EXPLICIT = 0.6

ROLE_FAMILIES = ["Software", "Data", "DevOps", "Security", "Product", "Management"]
SENIORITIES = ["Intern", "Junior", "Mid", "Senior", "Lead", "Manager"]
DOMAINS = ["FinTech", "E-commerce", "Healthcare", "Cyber", "SaaS", "Gaming"]

SENIORITY_WEIGHTS = [0.12, 0.22, 0.26, 0.20, 0.12, 0.08]

# Stats
STATS = Counter()
TOTAL_CALLS = 0

class ValidationError(Exception):
    def __init__(self, kind: str, skill: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.skill = skill


def sample_context() -> Dict[str, str]:
    seniority = random.choices(SENIORITIES, weights=SENIORITY_WEIGHTS, k=1)[0]

    if seniority in ["Intern", "Junior"]:
        responsibility_depth = "IC"
    elif seniority == "Mid":
        responsibility_depth = random.choice(["IC", "TechLead"])
    elif seniority in ["Senior", "Lead"]:
        responsibility_depth = random.choice(["IC", "TechLead"])
    else:
        responsibility_depth = "PeopleManager"

    return {
        "role_family": random.choice(ROLE_FAMILIES),
        "seniority": seniority,
        "responsibility_depth": responsibility_depth,
        "domain": random.choice(DOMAINS),
    }


# --------- IMPLICIT ALLOWED (Layer 2) ---------
IMPLICIT_ALLOWED = {
    "AWS EC2", "AWS Lambda", "AWS RDS", "AWS S3",
    "Azure Functions",
    "BigQuery", "Redshift", "Snowflake",
    "CI/CD", "GitOps", "GitHub Actions", "GitLab CI", "Jenkins",
    "Terraform", "CloudFormation",
    "Blue-Green Deployment", "Canary Releases", "ArgoCD",
    "Load Balancing", "Cloud Networking", "Security Hardening",
    "Performance Engineering",
    "Grafana", "Prometheus", "ELK Stack",
    "ETL", "ELT", "Airflow",
    "Parquet", "Avro", "Star Schema",
    "Distributed Systems", "Kafka",
    "MLOps", "MLflow",
    "Machine Learning", "NLP", "LLMs", "Transformers",
}

IMPLICIT_REPLACEMENTS = {
    "AWS Lambda": "event-driven serverless functions triggered by system events and queued messages",
    "GitOps": "declarative environment configuration synced from a repository with automated reconciliation",
    "CI/CD": "gated releases with automated checks before deploy and a rollback plan",
    "MLflow": "experiment tracking with a model registry and staged promotion between environments",
    "MLOps": "versioned training artifacts with automated validation checks before promotion",
    "Star Schema": "fact and dimension tables designed for analytics queries and reporting",
    "Cloud Networking": "subnet routing and security rules controlling connectivity between services",
    "Load Balancing": "request distribution across instances with health checks and failover routing",
    "Kafka": "event streams with producers/consumers, topic partitioning, and consumer groups",
    "Redshift": "a columnar data warehouse used for analytics with optimized reporting queries",
    "BigQuery": "serverless analytics queries over large datasets with partitioning-aware patterns",
    "Snowflake": "cloud data warehouse workflows with separated compute/storage for analytics",
    "Airflow": "scheduled DAG-based workflows with dependencies, retries, and backfills",
    "Parquet": "columnar storage files used to reduce size and speed up analytics reads",
    "Avro": "schema-based serialization for consistent data exchange between services",
    "Terraform": "infrastructure-as-code with planned changes, state management, and repeatable environments",
    "CloudFormation": "infrastructure templates defining resources and repeatable updates",
    "NLP": "text processing tasks like tokenization, normalization, and lightweight classification experiments",
}

IMPLICIT_HINT_PATTERNS = {
    "CI/CD": [
        r"\bci\s*/\s*cd\b",
        r"\bcontinuous integration\b",
        r"\bcontinuous deployment\b",
        r"\bdeployment gates?\b",
        r"\brelease gates?\b",
        r"\brollback\b",
        r"\brelease pipeline\b",
        r"\bdeployment pipeline\b",
        r"\bautomated test stages?\b",
        r"\b(build|test|deploy|release)\s+pipeline(s)?\b",
    ],
    "GitOps": [r"\bdesired state\b", r"\bautomated sync\b", r"\bdrift\b", r"\breconciliation\b"],
    "Kafka": [r"\btopic(s)?\b", r"\bpartition(s)?\b", r"\bconsumer group(s)?\b", r"\bproducer(s)?\b"],
    "MLflow": [r"\bmodel registry\b", r"\bexperiment tracking\b", r"\bpromotion\b"],
    "Machine Learning": [r"\btraining\b", r"\bclassification\b", r"\bmodel\b"],
}


def load_prompt_templates() -> List[str]:
    fp = os.path.join(PROMPTS_DIR, "unified_prompt.txt")
    if not os.path.exists(fp):
        # nicer error message
        existing = []
        if os.path.isdir(PROMPTS_DIR):
            existing = sorted(os.listdir(PROMPTS_DIR))
        raise RuntimeError(
            f"unified_prompt.txt not found in: {PROMPTS_DIR}\n"
            f"Existing files in Prompts: {existing}\n"
            f"Fix: make sure you have Prompts/unified_prompt.txt (exact name)."
        )
    with open(fp, "r", encoding="utf-8") as f:
        return [f.read()]


def build_skills_info(selected_skills: Dict[str, float]) -> Dict[str, Any]:
    # Keep this compact to reduce prompt tokens (important for cost + compliance)
    skills_info = {}
    for skill_name, label in selected_skills.items():
        aliases = SKILL_ALIASES.get(skill_name, {}).get("aliases", [])
        skills_info[skill_name] = {
            "target_label": label,
            "aliases": aliases,
        }
    return skills_info


def generate_prompt(selected_skills: Dict[str, float], template: str) -> str:
    skills_info = build_skills_info(selected_skills)
    skills_block = json.dumps(skills_info, indent=2, ensure_ascii=False)

    base_prompt = template.replace("[PASTE_SKILLS_DICT_HERE]", skills_block)

    ctx = sample_context()
    for k, v in ctx.items():
        base_prompt = base_prompt.replace("{" + k + "}", v)

    # Keep a small high-priority wrapper; the template already contains the detailed rules
    return f"""
Return ONLY ONE resume-style paragraph (4–6 sentences), plain text.
No JSON. No code. No markdown. No headings. No meta language.

{base_prompt}
""".strip()


def call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    global TOTAL_CALLS
    TOTAL_CALLS += 1

    client = OpenAI()

    # retry for transient errors / rate limits
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
        except Exception as e:
            # last attempt -> raise
            if attempt == 5:
                raise
            time.sleep(backoff)
            backoff = min(12.0, backoff * 2)


def _count_sentences(text: str) -> int:
    parts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    return len(parts)


def cleanup_after_deletions(text: str) -> str:
    t = text
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+,", ",", t)
    t = re.sub(r",\s*,", ",", t)
    t = re.sub(r"\band\s*,", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bof\s+and\b", "of", t, flags=re.IGNORECASE)
    t = re.sub(r"\(\s*\)", "", t)
    t = re.sub(r"\s+\.", ".", t)
    t = re.sub(r"\s+\!", "!", t)
    t = re.sub(r"\s+\?", "?", t)
    return t.strip()

STRICT_NONSELECTED_EXPLICIT = {
    "Machine Learning",
    "Load Balancing",
    "Distributed Systems",
    "CI/CD",
    "Cloud Networking",
}


def _split_sentences_keep_punct(text: str) -> List[str]:
    # Split into sentences while keeping punctuation end markers.
    # Example: ["Did X.", "Built Y!", "Tested Z?"]
    sents = re.findall(r"[^.!?]+[.!?]+|[^.!?]+$", text.strip())
    return [s.strip() for s in sents if s and s.strip()]

def trim_to_max_sentences(text: str, max_sentences: int = 6) -> str:
    sents = _split_sentences_keep_punct(text)
    if len(sents) <= max_sentences:
        return text.strip()
    trimmed = " ".join(sents[:max_sentences]).strip()
    # Ensure it ends with punctuation
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed

def _build_token_pattern(token: str) -> str:
    token = token.strip().lower()
    esc = re.escape(token)
    # If token is "word-like" -> use word boundaries
    if re.fullmatch(r"[a-z0-9][a-z0-9\.\+\-_/]*[a-z0-9]?", token):
        return rf"\b{esc}\b"
    # Otherwise fallback to non-word boundaries
    return rf"(?<!\w){esc}(?!\w)"

def validate_text(text: str, selected_skills: Dict[str, float]) -> str:
    if not text or len(text.strip()) < 30:
        raise ValidationError("text_invalid", "", "Text is empty/too short")

    text = text.strip()
    low = text.lower()

    banned_substrings = [
        "```", "import ", "def ", "class ", "print(", "json", "python script",
        "please provide", "based on the input json", "here is", "i'll", "i will",
        "corrected paragraph", "output json", "```python",
    ]
    if any(b in low for b in banned_substrings):
        raise ValidationError("text_invalid", "", "Meta/code/JSON detected in output")

    if any(ch in text for ch in ["{", "}", "[", "]"]):
        raise ValidationError("text_invalid", "", "Braces/brackets detected (likely JSON/code)")

    # ✅ Sentence count: trim instead of FAIL when too many
    n_sent = _count_sentences(text)
    if n_sent > 6:
        text = trim_to_max_sentences(text, 6)
        low = text.lower()
        n_sent = _count_sentences(text)

    if n_sent < 4:
        raise ValidationError("text_invalid", "", f"Too few sentences: {n_sent} (expected 4–6)")

    # Explicit / implicit checks for SELECTED skills
    for skill_name, label in selected_skills.items():
        canon = skill_name.lower()
        aliases = [a.lower() for a in SKILL_ALIASES.get(skill_name, {}).get("aliases", [])]

        if label == 1.0:
            if not (canon in low or any(a in low for a in aliases)):
                raise ValidationError("explicit_missing", skill_name, f"Explicit skill missing: {skill_name}")
        elif label == 0.5:
            if canon in low or any(a in low for a in aliases):
                raise ValidationError("implicit_leaked", skill_name, f"Implicit skill leaked: {skill_name}")

    selected_set = set(selected_skills.keys())

    # ✅ Non-selected explicit mentions: only enforce strict set (NOT all skills)
    # This prevents constant failures on generic words like "Azure", "cloud", etc.
    strict_targets = [s for s in STRICT_NONSELECTED_EXPLICIT if s not in selected_set]

    for other_skill in strict_targets:
        info = SKILL_ALIASES.get(other_skill, {})
        candidates = [other_skill] + info.get("aliases", [])
        for cand in candidates:
            if not cand:
                continue
            token = cand.strip().lower()
            # Skip ultra-short / noisy tokens
            if re.fullmatch(r"[a-z]+", token) and len(token) <= 2:
                continue

            pat = _build_token_pattern(token)
            if re.search(pat, low):
                # Special case: Machine Learning is super common -> treat as strict (you chose it),
                # so still FAIL here. If later you want "sanitize not fail", tell me and I'll adjust.
                raise ValidationError("nonselected_explicit", other_skill, f"Mentions non-selected STRICT skill: {other_skill}")

    # ✅ Non-selected implied: only enforce a small strict implied set (as you already do)
    STRICT_NONSELECTED_IMPLIED = {"CI/CD", "GitOps", "MLflow", "Machine Learning"}
    for hinted_skill, pats in IMPLICIT_HINT_PATTERNS.items():
        if hinted_skill not in STRICT_NONSELECTED_IMPLIED:
            continue
        if hinted_skill in selected_set:
            continue
        for pat in pats:
            if re.search(pat, low):
                raise ValidationError("nonselected_implied", hinted_skill, f"Implies non-selected skill: {hinted_skill}")

    return text


def pick_random_skills(k_min: int = 3, k_max: int = 6) -> Dict[str, float]:
    k = random.randint(k_min, k_max)
    selected_names = random.sample(GLOBAL_SKILL_VECTOR, k)

    skills_with_labels = {}
    for name in selected_names:
        category = SKILL_ALIASES.get(name, {}).get("category", "")

        # Programming languages cannot be implicit
        if category == "programming_language":
            skills_with_labels[name] = 1.0
            continue

        if name not in IMPLICIT_ALLOWED:
            skills_with_labels[name] = 1.0
        else:
            skills_with_labels[name] = 1.0 if random.random() < P_EXPLICIT else 0.5

    return skills_with_labels


def force_add_missing_explicit(text: str, missing_skill: str) -> str:
    # IMPORTANT: do NOT use the banned phrase ("using <skill> in day-to-day...")
    t = text.strip()
    if not t:
        return f"Delivered project work with {missing_skill}."

    m = re.search(r"^(.*?)([.!?])\s*$", t)
    if not m:
        return (t + f", with {missing_skill} applied to implementation and maintenance.").strip()

    body, punct = m.group(1), m.group(2)
    if missing_skill.lower() in t.lower():
        return t

    injected = f"{body}, with {missing_skill} applied to implementation and maintenance{punct}"
    injected = re.sub(r"\s{2,}", " ", injected).strip()
    return injected


def sanitize_implicit_leak(text: str, leaked_skill: str) -> str:
    aliases = SKILL_ALIASES.get(leaked_skill, {}).get("aliases", [])
    repl = IMPLICIT_REPLACEMENTS.get(leaked_skill, "production-grade engineering work")
    candidates = [leaked_skill] + aliases

    for cand in sorted([c for c in candidates if c], key=len, reverse=True):
        token = cand.strip()
        if not token:
            continue

        # Works for single-word and multi-word skills (e.g., "AWS EC2")
        pat = r"(?i)(?<!\w)" + re.escape(token) + r"(?!\w)"
        text = re.sub(pat, repl, text)

    return cleanup_after_deletions(text)



def enforce_no_as_a_opening(text: str, allow_prob: float = 0.1) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if stripped.lower().startswith("as a"):
        if random.random() > allow_prob:
            text = re.sub(r"(?i)^\s*as a[^,.]*[,.]\s*", "", text).strip()
    return text


def _post_process(text: str) -> str:
    text = enforce_no_as_a_opening(text, allow_prob=0.1)
    text = cleanup_after_deletions(text)
    return text


def build_fix_prompt(original_text: str, selected_skills: Dict[str, float]) -> str:
    skills_info = build_skills_info(selected_skills)
    skills_block = json.dumps(skills_info, indent=2, ensure_ascii=False)

    return f"""
Return ONLY the corrected paragraph (plain text), 4–6 sentences.
No JSON. No code. No markdown. No headings. No meta text.

Rules:
- label 1.0: MUST mention canonical skill name OR one alias verbatim.
- label 0.5: MUST NOT mention canonical name NOR any alias; imply via responsibilities.
- Do NOT add any skills beyond the list.

skills_info:
{skills_block}

Original paragraph:
{original_text}
""".strip()


def generate_sample(templates: List[str]) -> Optional[Dict[str, Any]]:
    selected_skills = pick_random_skills()
    template = random.choice(templates)
    prompt = generate_prompt(selected_skills, template)

    last_err: Optional[Exception] = None
    text = ""

    # 1) initial generation attempts
    for _ in range(MAX_GEN_RETRIES):
        try:
            text = call_openai(prompt, TEMPERATURE_GEN, MAX_TOKENS_GEN)
            text = _post_process(text)
            text = validate_text(text, selected_skills)  # ✅ validate the right variable
            STATS["pass_initial"] += 1
            return {"job_description": text, "skills": selected_skills}
        except ValidationError as ve:
            last_err = ve
        except Exception as e:
            last_err = e

    # 2) local repair loop
    repaired_text = text
    for _ in range(3):
        try:
            repaired_text = _post_process(repaired_text)
            repaired_text = validate_text(repaired_text, selected_skills)  # ✅ validate repaired_text
            STATS["pass_repair"] += 1
            return {"job_description": repaired_text, "skills": selected_skills}
        except ValidationError as ve:
            if ve.kind == "explicit_missing" and ve.skill:
                STATS["repair_add_explicit"] += 1
                repaired_text = force_add_missing_explicit(repaired_text, ve.skill)
                repaired_text = cleanup_after_deletions(repaired_text)
                continue
            if ve.kind == "implicit_leaked" and ve.skill:
                STATS["repair_sanitize_implicit"] += 1
                repaired_text = sanitize_implicit_leak(repaired_text, ve.skill)
                repaired_text = cleanup_after_deletions(repaired_text)
                continue
            last_err = ve
            break
        except Exception as e:
            last_err = e
            break

    # 3) LLM fix passes
    fix_prompt = build_fix_prompt(repaired_text, selected_skills)
    fixed_text = repaired_text

    for _ in range(MAX_FIX_RETRIES):
        try:
            fixed_text = call_openai(fix_prompt, TEMPERATURE_FIX, MAX_TOKENS_FIX)
            fixed_text = _post_process(fixed_text)
            fixed_text = validate_text(fixed_text, selected_skills)  # ✅ validate fixed_text
            STATS["pass_fix_llm"] += 1
            return {"job_description": fixed_text, "skills": selected_skills}
        except ValidationError as ve:
            last_err = ve
            if ve.kind == "explicit_missing" and ve.skill:
                STATS["fix_add_explicit"] += 1
                fixed_text = force_add_missing_explicit(fixed_text, ve.skill)
                fix_prompt = build_fix_prompt(fixed_text, selected_skills)
                continue
            if ve.kind == "implicit_leaked" and ve.skill:
                STATS["fix_sanitize_implicit"] += 1
                fixed_text = sanitize_implicit_leak(fixed_text, ve.skill)
                fix_prompt = build_fix_prompt(fixed_text, selected_skills)
                continue
        except Exception as e:
            last_err = e
            break

    STATS["skip"] += 1
    msg = str(last_err) if last_err else "unknown"
    print(f"[SKIP] Failed. Reason: {msg}")
    return None



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    templates = load_prompt_templates()
    
    # Get next versioned output file
    output_file = get_next_versioned_output_file()

    print(f"Model: {OPENAI_MODEL}")
    print(f"Prompts dir: {PROMPTS_DIR}")
    print(f"Output: {output_file}")
    print(f"Loaded {len(templates)} templates")

    generated = 0
    with open(output_file, "w", encoding="utf-8") as f:
        pbar = tqdm(total=NUM_SAMPLES)
        while generated < NUM_SAMPLES:
            sample = generate_sample(templates)
            if sample:
                json.dump(sample, f, ensure_ascii=False)
                f.write("\n")
                generated += 1
                pbar.update(1)
        pbar.close()

    print(f"Done. Wrote {generated} samples to {output_file}")

    print("\n--- STATS ---")
    print(f"Total model calls: {TOTAL_CALLS}")
    print(f"Success rate: {generated}/{max(1, TOTAL_CALLS)} = {generated / max(1, TOTAL_CALLS):.2%}")
    for k, v in STATS.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
