# EDA/eda_extensions.py
from __future__ import annotations

import os
import sys
import json
import re
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt

# ----------------------------
# Make project root importable
# ----------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---- project imports ----
from skills.skillAliases import skills as SKILL_ALIASES  # canonical -> {category, aliases}
from skills.globalVector import GLOBAL_SKILL_VECTOR      # sorted list of all skills
# -------------------------


# Optional heavy deps
USE_PANDAS = True
USE_SKLEARN = True

try:
    import pandas as pd
except Exception:
    USE_PANDAS = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    USE_SKLEARN = False


# ----------------------------
# Helpers
# ----------------------------
def _normalize_text(t: str) -> str:
    return (t or "").strip()


def _word_boundary_pattern(phrase: str) -> re.Pattern:
    """
    Whole-token boundary match, case-insensitive.
    """
    esc = re.escape((phrase or "").lower())
    return re.compile(rf"(?<!\w){esc}(?!\w)")


def _build_alias_patterns() -> Dict[str, List[re.Pattern]]:
    """
    For each canonical skill build patterns for canonical + aliases with word boundaries.
    Filters out extremely short aliases (<2 chars) to reduce false positives.
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


# ----------------------------
# Data loading
# ----------------------------
def load_jsonl(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = obj.get("job_description") or obj.get("resume_chunk_text") or obj.get("text")
            if text is None:
                continue

            skills = obj.get("skills", {})
            if not isinstance(skills, dict):
                continue

            rows.append(
                {
                    "idx": i,
                    "job_description": _normalize_text(text),
                    "skills": skills,  # dict skill -> {1.0,0.5}
                }
            )
    return rows


# ----------------------------
# Stats from data
# ----------------------------
def compute_skill_stats(rows: List[Dict[str, Any]]):
    overall = Counter()
    explicit = Counter()
    implicit = Counter()

    per_sample_counts: List[int] = []
    per_sample_explicit: List[int] = []
    per_sample_implicit: List[int] = []

    pair_counts = Counter()
    unknown_skills = Counter()

    global_set = set(GLOBAL_SKILL_VECTOR)

    for r in rows:
        skills = r["skills"]
        ks = sorted(skills.keys())

        # co-occurrence
        for a, b in combinations(ks, 2):
            pair_counts[(a, b)] += 1

        exp = 0
        imp = 0
        for s, v in skills.items():
            if s not in global_set:
                unknown_skills[s] += 1

            overall[s] += 1
            if float(v) == 1.0:
                explicit[s] += 1
                exp += 1
            elif float(v) == 0.5:
                implicit[s] += 1
                imp += 1

        per_sample_counts.append(len(skills))
        per_sample_explicit.append(exp)
        per_sample_implicit.append(imp)

    return {
        "overall": overall,
        "explicit": explicit,
        "implicit": implicit,
        "pair_counts": pair_counts,
        "per_sample_counts": per_sample_counts,
        "per_sample_explicit": per_sample_explicit,
        "per_sample_implicit": per_sample_implicit,
        "unknown_skills": unknown_skills,
    }


def build_global_counts(stats_dict: dict) -> List[Tuple[str, int]]:
    """
    Return list of (skill, count) for ALL skills in GLOBAL_SKILL_VECTOR.
    Missing skills will have count=0.
    """
    overall: Counter = stats_dict["overall"]
    return [(s, int(overall.get(s, 0))) for s in GLOBAL_SKILL_VECTOR]


def exact_duplicates(rows: List[Dict[str, Any]]):
    """
    Exact duplicates by identical job_description string.
    """
    hashes = Counter()
    by_hash = defaultdict(list)

    for r in rows:
        t = r["job_description"]
        h = hashlib.md5(t.encode("utf-8")).hexdigest()
        hashes[h] += 1
        by_hash[h].append(r["idx"])

    dup_hashes = {h: c for h, c in hashes.items() if c > 1}
    return dup_hashes, by_hash


def implicit_leak_check(rows: List[Dict[str, Any]], alias_patterns: Dict[str, List[re.Pattern]]):
    """
    For each implicit-labeled skill: check if canonical/aliases appear in text.
    Returns leak count + example leaks.
    """
    leaks = []
    leak_count = 0

    for r in rows:
        text_low = r["job_description"].lower()
        for skill, v in r["skills"].items():
            if float(v) != 0.5:
                continue

            pats = alias_patterns.get(skill, [])
            hit = any(p.search(text_low) for p in pats)

            if hit:
                leak_count += 1
                leaks.append(
                    {
                        "idx": r["idx"],
                        "skill": skill,
                        "snippet": r["job_description"][:180].replace("\n", " "),
                    }
                )

    return leak_count, leaks


def near_duplicates_tfidf(rows: List[Dict[str, Any]], threshold: float = 0.93, top_k: int = 30):
    """
    Near-duplicate detection using TF-IDF cosine similarity.
    Requires scikit-learn.
    """
    if not USE_SKLEARN:
        return []

    texts = [r["job_description"] for r in rows]
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X = vec.fit_transform(texts)
    sims = cosine_similarity(X, dense_output=False).tocoo()

    pairs = []
    for i, j, v in zip(sims.row, sims.col, sims.data):
        if i >= j:
            continue
        if float(v) >= threshold:
            pairs.append((i, j, float(v)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_k]

    out = []
    for i, j, sim in pairs:
        out.append(
            {
                "idx1": rows[i]["idx"],
                "idx2": rows[j]["idx"],
                "cosine_sim": round(sim, 4),
                "snippet1": rows[i]["job_description"][:140].replace("\n", " "),
                "snippet2": rows[j]["job_description"][:140].replace("\n", " "),
            }
        )
    return out


def category_breakdown_from_global(stats_dict: dict):
    """
    Category totals based on GLOBAL vector counts (including 0-occ).
    Also returns category coverage (how many skills in that category are missing).
    """
    overall: Counter = stats_dict["overall"]
    explicit: Counter = stats_dict["explicit"]
    implicit: Counter = stats_dict["implicit"]

    cat_total = Counter()
    cat_explicit = Counter()
    cat_implicit = Counter()

    cat_skill_total = Counter()     # how many skills exist in category (from global)
    cat_skill_missing = Counter()   # how many skills have 0 occurrences

    for skill in GLOBAL_SKILL_VECTOR:
        meta = SKILL_ALIASES.get(skill, {})
        cat = meta.get("category", "UNKNOWN")

        c_total = int(overall.get(skill, 0))
        c_exp = int(explicit.get(skill, 0))
        c_imp = int(implicit.get(skill, 0))

        cat_total[cat] += c_total
        cat_explicit[cat] += c_exp
        cat_implicit[cat] += c_imp

        cat_skill_total[cat] += 1
        if c_total == 0:
            cat_skill_missing[cat] += 1

    return cat_total, cat_explicit, cat_implicit, cat_skill_total, cat_skill_missing


# ----------------------------
# Save outputs (GLOBAL-aware)
# ----------------------------
def save_tables(out_dir: Path, rows: List[Dict[str, Any]], stats_dict: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    n_samples = max(1, len(rows))
    overall: Counter = stats_dict["overall"]
    explicit: Counter = stats_dict["explicit"]
    implicit: Counter = stats_dict["implicit"]

    # Build GLOBAL-aware table
    global_counts = [(s, int(overall.get(s, 0))) for s in GLOBAL_SKILL_VECTOR]

    if USE_PANDAS:
        df = pd.DataFrame(global_counts, columns=["skill", "count"])
        df["explicit"] = df["skill"].map(lambda s: int(explicit.get(s, 0)))
        df["implicit"] = df["skill"].map(lambda s: int(implicit.get(s, 0)))
        df["pct_of_samples"] = (df["count"] / n_samples * 100).round(2)
        df["missing_0occ"] = df["count"].map(lambda c: c == 0)

        # Full table (ALL skills)
        df.to_csv(out_dir / "skills_global_counts.csv", index=False)

        # Missing skills only
        df_missing = df[df["count"] == 0].copy()
        df_missing.to_csv(out_dir / "missing_skills_0occ.csv", index=False)

        # Top 15 (count>0)
        df_top = df[df["count"] > 0].sort_values("count", ascending=False).head(15)
        df_top.to_csv(out_dir / "top15_skills_global.csv", index=False)

        # Worst 15 (including 0)
        df_worst = df.sort_values("count", ascending=True).head(15)
        df_worst.to_csv(out_dir / "worst15_skills_global.csv", index=False)

        # Rare skills (bottom 30 including 0)
        df_rare30 = df.sort_values("count", ascending=True).head(30)
        df_rare30.to_csv(out_dir / "rare30_skills_global.csv", index=False)

        # Top co-occurrence pairs (from data)
        pair_counts = stats_dict["pair_counts"].most_common(50)
        df_pairs = pd.DataFrame(
            [{"skill_a": a, "skill_b": b, "count": c} for (a, b), c in pair_counts]
        )
        df_pairs.to_csv(out_dir / "top_pairs.csv", index=False)

        # Unknown skills (if any)
        unknown = stats_dict.get("unknown_skills", Counter())
        if unknown:
            df_unknown = pd.DataFrame(unknown.most_common(), columns=["skill", "count"])
            df_unknown.to_csv(out_dir / "unknown_skills_in_data.csv", index=False)


def save_plots(out_dir: Path, rows: List[Dict[str, Any]], stats_dict: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    overall: Counter = stats_dict["overall"]

    # GLOBAL-aware counts list
    all_counts = [(s, int(overall.get(s, 0))) for s in GLOBAL_SKILL_VECTOR]

    # Top 15 (count>0)
    top_candidates = [(s, c) for s, c in all_counts if c > 0]
    top15 = sorted(top_candidates, key=lambda x: x[1], reverse=True)[:15]

    if top15:
        skills = [s for s, _ in reversed(top15)]
        counts = [c for _, c in reversed(top15)]

        plt.figure(figsize=(10, 6))
        plt.barh(skills, counts)
        plt.title("Top 15 skills (from Global Vector, count>0)")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / "top15_skills_global.png", dpi=200)
        plt.close()
    else:
        print("WARNING: No skills with count>0 found (dataset may be empty).")

    # Worst 15 (including 0-occ)
    worst15 = sorted(all_counts, key=lambda x: x[1])[:15]
    w_skills = [s for s, _ in worst15]
    w_counts = [c for _, c in worst15]

    plt.figure(figsize=(10, 6))
    plt.barh(w_skills, w_counts)
    plt.title("Worst 15 skills (from Global Vector, including 0-occ)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "worst15_skills_global.png", dpi=200)
    plt.close()

    # Histogram: implicit per sample
    imp = stats_dict["per_sample_implicit"]
    if imp:
        plt.figure(figsize=(10, 6))
        plt.hist(imp, bins=range(0, max(imp) + 2))
        plt.title("Histogram: #Implicit labels per sample")
        plt.xlabel("#Implicit in sample")
        plt.ylabel("Samples")
        plt.tight_layout()
        plt.savefig(out_dir / "implicit_per_sample_hist.png", dpi=200)
        plt.close()

    # Category plots (GLOBAL-aware)
    cat_total, cat_exp, cat_imp, cat_skill_total, cat_skill_missing = category_breakdown_from_global(stats_dict)

    cats = [c for c, _ in cat_total.most_common()]
    totals = [cat_total[c] for c in cats]

    plt.figure(figsize=(12, 6))
    plt.bar(cats, totals)
    plt.title("Label count by category (GLOBAL-aware, includes 0-occ skills)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "category_total_global.png", dpi=200)
    plt.close()

    # Implicit share per category (GLOBAL-aware)
    shares = []
    for c in cats:
        total = cat_total[c]
        shares.append((cat_imp[c] / total) if total else 0.0)

    plt.figure(figsize=(12, 6))
    plt.bar(cats, shares)
    plt.title("Implicit share by category (GLOBAL-aware)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Implicit / Total")
    plt.tight_layout()
    plt.savefig(out_dir / "category_implicit_share_global.png", dpi=200)
    plt.close()

    # Category coverage: missing skills ratio per category
    cov_vals = []
    for c in cats:
        total_sk = cat_skill_total[c]
        miss_sk = cat_skill_missing[c]
        cov_vals.append((miss_sk / total_sk) if total_sk else 0.0)

    plt.figure(figsize=(12, 6))
    plt.bar(cats, cov_vals)
    plt.title("Missing skills ratio by category (0-occ / total skills in category)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Missing ratio")
    plt.tight_layout()
    plt.savefig(out_dir / "category_missing_ratio.png", dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    dataset_path = Path("data/synthetic_dataset.jsonl")
    out_dir = Path("outputs/eda_extensions")

    rows = load_jsonl(dataset_path)
    print("Loaded samples:", len(rows))

    stats_dict = compute_skill_stats(rows)

    # GLOBAL coverage summary
    all_counts = build_global_counts(stats_dict)
    missing = [s for s, c in all_counts if c == 0]
    print(f"Global skills: {len(GLOBAL_SKILL_VECTOR)}")
    print(f"Missing skills (0 occurrences): {len(missing)}")

    # Exact duplicates
    dup_hashes, by_hash = exact_duplicates(rows)
    dup_total_extra = sum(c - 1 for c in dup_hashes.values())
    print("Exact duplicate texts (extra rows):", dup_total_extra)

    # Implicit leak check (canonical+aliases)
    alias_patterns = _build_alias_patterns()
    leak_count, leaks = implicit_leak_check(rows, alias_patterns)
    print("Implicit leak hits (canonical/alias found in text):", leak_count)

    # Near duplicates (optional)
    near = near_duplicates_tfidf(rows, threshold=0.93, top_k=30) if USE_SKLEARN else []
    print("Near-duplicate pairs (tf-idf >= 0.93):", len(near))

    # Save tables + plots
    save_tables(out_dir, rows, stats_dict)
    save_plots(out_dir, rows, stats_dict)

    # Save duplicates/leaks/near examples
    if USE_PANDAS:
        if dup_hashes:
            df_dups = pd.DataFrame(
                [{"hash": h, "count": c, "sample_idxs": by_hash[h]} for h, c in sorted(dup_hashes.items(), key=lambda x: -x[1])]
            )
            df_dups.to_csv(out_dir / "exact_duplicates.csv", index=False)

        if leaks:
            pd.DataFrame(leaks[:100]).to_csv(out_dir / "implicit_leaks_examples.csv", index=False)

        if near:
            pd.DataFrame(near).to_csv(out_dir / "near_duplicates.csv", index=False)

        # Also save missing list as plain txt for quick glance
        (out_dir / "missing_skills_0occ.txt").write_text("\n".join(missing), encoding="utf-8")

    print("Saved outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
