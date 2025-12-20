# skills/implicitCueBank.py
# Hybrid evidence system for IMPLICIT (0.5):
# - Prefer skill-specific evidence phrases (e.g., NLP, MLOps) when available.
# - Otherwise use category evidence phrases.
# - Evidence phrases must NOT include canonical skill names or aliases (no leak),
#   and should avoid naming other skills from your global vector.

from __future__ import annotations

import random
import re
from typing import Dict, List, Any, Optional

# -----------------------------
# IMPLICIT eligibility policy
# -----------------------------

# Categories where category-level evidence is OK (not vendor/tool-specific).
SAFE_CATEGORY_IMPLICIT = {
    "security",
    "testing",
    "monitoring_logging",
    "systems_design",
    "architecture",
    "methodology",
    "project_management",
    "version_control",
    "ml_concept",
}

# Categories where you should require SKILL-specific evidence if implicit
# (typically tools/vendors/frameworks).
TOOL_LIKE_CATEGORIES = {
    "database",
    "data_warehouse",
    "cloud_platform",
    "ml_framework",
    "ml_library",
    "security_tool",
    "ci_cd_tool",
    "devops_tool",
    "data_engineering",
}

# Minimum number of skill-specific evidence phrases to consider a tool/vendor skill implicit-eligible.
MIN_SKILL_EVIDENCE_PHRASES = 6


# -----------------------------
# Variety helpers (for "implicit_cues" text generation)
# -----------------------------
METRICS = [
    "reduced p95 latency", "reduced p99 latency", "reduced error rate", "improved uptime",
    "improved MTTR", "improved throughput", "reduced deployment time", "reduced regression rate",
    "reduced cost", "reduced query time", "reduced memory footprint", "improved CPU utilization",
    "improved reliability", "improved auditability", "improved traceability", "improved data quality",
    "improved release confidence", "improved scalability", "improved security posture",
]

SCALE_CONTEXT = [
    "during peak traffic", "under bursty workloads", "at high concurrency", "across multiple regions",
    "with multi-tenant constraints", "under strict compliance requirements", "in a production environment",
    "with zero-downtime expectations", "under tight SLAs", "for latency-sensitive workloads",
]

CONSTRAINTS = [
    "with strict SLAs", "with limited maintenance windows", "with backwards compatibility requirements",
    "while minimizing downtime", "while avoiding breaking changes", "while keeping costs predictable",
    "while meeting audit requirements", "while handling partial failures", "while preserving data consistency",
]

IMPACT_FRAMES = [
    "to {metric} {scale}",
    "resulting in {metric} {scale}",
    "which helped {metric} {scale}",
    "leading to {metric} {scale}",
]

STYLE_HINTS = [
    "Keep the tone concise and technical.",
    "Use specific outcomes and production-grade phrasing.",
    "Focus on reliability, validation, and measurable improvements.",
    "Avoid buzzwords; describe concrete responsibilities.",
]

def pick_metric(rng: random.Random) -> str:
    return rng.choice(METRICS)

def pick_scale(rng: random.Random) -> str:
    return rng.choice(SCALE_CONTEXT)

def build_impact(rng: random.Random) -> str:
    metric = pick_metric(rng)
    scale = pick_scale(rng)
    frame = rng.choice(IMPACT_FRAMES)
    return frame.format(metric=metric, scale=scale)

CATEGORY_CUES: Dict[str, Dict[str, List[str]]] = {
    "security": {
        "actions": ["enforced", "introduced", "standardized", "hardened", "reviewed"],
        "objects": ["least-privilege access controls", "credential rotation", "policy checks", "denylist rules", "audit trails"],
    },
    "testing": {
        "actions": ["implemented", "refined", "expanded", "stabilized", "automated"],
        "objects": ["contract checks", "negative test cases", "status and payload assertions", "test fixtures", "test doubles"],
    },
    "monitoring_logging": {
        "actions": ["instrumented", "monitored", "triaged", "standardized", "alerted on"],
        "objects": ["audit logs", "service metrics", "SLA dashboards", "error-rate alerts", "latency histograms"],
    },
    "systems_design": {
        "actions": ["designed", "implemented", "validated", "optimized", "rolled out"],
        "objects": ["retry and rollback mechanisms", "idempotency enforcement", "rate limiting protections", "backpressure controls", "graceful degradation"],
    },
    "version_control": {
        "actions": ["managed", "reviewed", "standardized", "enforced", "automated"],
        "objects": ["branch protections", "code review workflows", "release tagging conventions", "merge checks", "CI status gates"],
    },
    "project_management": {
        "actions": ["tracked", "coordinated", "planned", "aligned", "reported"],
        "objects": ["milestones", "stakeholder updates", "incident postmortems", "risk registers", "delivery timelines"],
    },
    "ml_concept": {
        "actions": ["evaluated", "monitored", "validated", "analyzed", "reduced"],
        "objects": ["data drift", "feature leakage", "false positives and negatives", "model regressions", "offline evaluation metrics"],
    },
}

# -----------------------------
# Evidence phrase banks (validated for IMPLICIT)
# -----------------------------

# Skill-specific evidence phrases (for ambiguous skills like NLP / MLOps).
SKILL_EVIDENCE_PHRASES: Dict[str, List[str]] = {
    # NLP (DO NOT use "nlp" or "natural language processing")
   "NLP": [
        "tokenization",
        "text normalization",
        "stopword removal",
        "stemming",
        "lemmatization",
        "named entity recognition",
        "part-of-speech tagging",
        "sequence labeling",
        "n-gram features",
        "text classification",
        "sentence segmentation",
        "spelling normalization",
        "regex-based preprocessing",
        "keyword extraction",
    ],

    # MLOps (DO NOT use "mlops" / "ml ops")
    "MLOps": [
        "model versioning",
        "model registry",
        "staged promotion",
        "drift monitoring",
        "retryable pipelines",
        "artifact tracking",
        "offline evaluation",
        "data drift symptoms",
        "feature leakage checks",
        "rollback to previous model",
    ],

    # Transformers (avoid 'transformers')
    "Transformers": [
        "attention-based encoders",
        "sequence-to-sequence finetuning",
        "token classification heads",
        "inference batching",
        "context-window constraints",
        "prompt templating",
        "embedding-based retrieval",
    ],

    # dbt (avoid 'dbt' / 'data build tool')
    "dbt": [
        "incremental models",
        "model refactoring",
        "source freshness tests",
        "data lineage documentation",
        "staging models",
        "schema tests",
        "seed files",
        "environment-specific targets",
    ],

    # Postman (avoid 'postman')
    "Postman": [
        "API collections",
        "collection runner",
        "pre-request scripts",
        "environment variables for requests",
        "response assertions",
        "mock servers for APIs",
    ],

    # Jenkins (avoid 'jenkins')
    "Jenkins": [
        "pipeline stages",
        "build agents",
        "artifact publishing",
        "job orchestration",
        "build logs triage",
        "release tagging conventions",
    ],

    # Snowflake (avoid 'snowflake')
    "Snowflake": [
        "virtual warehouses",
        "separate compute and storage",
        "zero-copy cloning",
        "time travel for tables",
        "auto-suspend warehouses",
    ],

    # BigQuery (avoid 'bigquery')
    "BigQuery": [
        "partitioned tables",
        "clustering keys",
        "scheduled queries",
        "slot reservations",
        "federated queries",
    ],

    # scikit-learn (avoid 'scikit-learn' / 'sklearn')
    "scikit-learn": [
        "grid search",
        "pipeline fit/predict",
        "classification metrics",
        "cross-validation folds",
    ],
    "LLMs": [
        "prompt templating",
        "context-window constraints",
        "inference batching",
        "guardrail rules",
        "structured outputs",
        "evaluation rubrics",
    ],

    # IAM (avoid 'iam' / 'identity and access management')
    "IAM": [
        "least-privilege access controls",
        "role-based access control",
        "policy-based access enforcement",
        "scoped permissions",
        "service accounts",
        "permission audits",
        "privileged access reviews",
        "permission boundaries",
    ],
}

CATEGORY_EVIDENCE_PHRASES: Dict[str, List[str]] = {
    "security": [
        "least-privilege access controls",
        "credential rotation",
        "policy checks",
        "audit trails",
        "denylist rules",
    ],
    "testing": [
        "contract checks",
        "negative test cases",
        "status and payload assertions",
        "test fixtures",
        "test doubles",
    ],
    "monitoring_logging": [
        "audit logs",
        "service metrics",
        "error-rate alerts",
        "latency histograms",
        "SLA dashboards",
    ],
    "systems_design": [
        "retry and rollback mechanisms",
        "idempotency enforcement",
        "rate limiting protections",
        "backpressure controls",
        "graceful degradation",
    ],
    "version_control": [
        "branch protections",
        "code review workflows",
        "merge checks",
        "CI status gates",
    ],
    "project_management": [
        "stakeholder updates",
        "delivery timelines",
        "incident postmortems",
        "risk registers",
    ],
    "ml_concept": [
        "data drift",
        "feature leakage checks",
        "false positives and negatives",
        "offline evaluation",
    ],
}

def style_hint(rng: random.Random) -> str:
    return rng.choice(STYLE_HINTS)

# Optional overrides for skills whose category in skillAliases is too generic/incorrect for evidence selection.
SKILL_CATEGORY_OVERRIDES: Dict[str, str] = {
    "IAM": "security_tool",
}

def get_category(skill_aliases: Dict[str, Any], skill_name: str) -> Optional[str]:
    return SKILL_CATEGORY_OVERRIDES.get(skill_name) or (skill_aliases.get(skill_name) or {}).get("category")

def category_has_evidence(category: Optional[str]) -> bool:
    if not category:
        return False
    return category in CATEGORY_EVIDENCE_PHRASES and len(CATEGORY_EVIDENCE_PHRASES[category]) > 0

def skill_has_evidence(skill_name: str) -> bool:
    phrases = SKILL_EVIDENCE_PHRASES.get(skill_name, [])
    return len(phrases) >= MIN_SKILL_EVIDENCE_PHRASES

def skill_or_category_has_evidence(skill_name: str, category: Optional[str]) -> bool:
    # For tool-like categories, require skill-specific evidence (to avoid ambiguity).
    if category in TOOL_LIKE_CATEGORIES:
        return skill_has_evidence(skill_name)

    # For safe categories, allow either skill-specific evidence or category evidence.
    if skill_has_evidence(skill_name):
        return True
    return category_has_evidence(category)

def pick_required_evidence_phrases(
    rng: random.Random,
    skill_name: str,
    category: Optional[str],
    k_min: int = 1,
    k_max: int = 2,
) -> List[str]:
    # Prefer skill-specific evidence
    phrases = SKILL_EVIDENCE_PHRASES.get(skill_name, [])
    if len(phrases) >= MIN_SKILL_EVIDENCE_PHRASES:
        k = rng.randint(k_min, min(k_max, len(phrases)))
        return rng.sample(phrases, k)

    # Fallback to category-level evidence if allowed
    cat_phrases = CATEGORY_EVIDENCE_PHRASES.get(category or "", [])
    if cat_phrases:
        k = rng.randint(k_min, min(k_max, len(cat_phrases)))
        return rng.sample(cat_phrases, k)

    return []

def text_contains_any_required_phrase(text: str, required_phrases: List[str]) -> bool:
    low = text.lower()
    for p in required_phrases:
        if p.lower() in low:
            return True
    return False

def build_dynamic_implicit_sentence(
    rng: random.Random,
    category: str,
    skill_name: str,
) -> str:
    cues = CATEGORY_CUES.get(category)
    impact = build_impact(rng)

    if not cues:
        fallbacks = [
            f"Implemented concrete responsibilities and outcomes {impact}.",
            f"Designed operational improvements and validation steps {impact}.",
            f"Hardened critical workflows with reliability and safety controls {impact}.",
        ]
        return rng.choice(fallbacks)

    action = rng.choice(cues["actions"])
    obj = rng.choice(cues["objects"])
    constraint = rng.choice(CONSTRAINTS)

    # Keep it generic; do not name the skill.
    sent = f"{action.capitalize()} {obj} {constraint}, {impact}."
    # Trim double spaces
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent
