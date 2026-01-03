#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EDA for the synthetic skill-mapping dataset.

Reads:
- data/plans/plans_v1.jsonl              (if exists)
- data/synthetic_dataset_extra.jsonl     (required)
- src/skills/skills_v1.txt               (optional, for total #skills)

Outputs:
- Console summary (for report/slide)
- Simple plots saved to: data/outputs/EDA/
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ---------- paths ----------

SCRIPT_DIR = Path(__file__).resolve().parent        # .../src/EDA
PROJECT_ROOT = SCRIPT_DIR.parent.parent             # project root
DATA_DIR = PROJECT_ROOT / "data"

OUT_DIR = DATA_DIR / "outputs" / "EDA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLANS_PATH = DATA_DIR / "plans" / "plans_v1.jsonl"                     # data/plans/plans_v1.jsonl
DATASET_PATH = DATA_DIR / "synthetic_dataset_extra.jsonl"              # *** חדש ***
SKILLS_PATH = PROJECT_ROOT / "src" / "skills" / "skills_v1.txt"        # src/skills/skills_v1.txt


# ---------- helpers ----------

def load_jsonl(path: Path):
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse line in {path.name}: {e}")
    print(f"[INFO] Loaded {len(rows)} rows from {path.name}")
    return rows


def load_skills(path: Path):
    if not path.exists():
        print(f"[WARN] Skills file not found: {path}")
        return []
    skills = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            skills.append(line)
    print(f"[INFO] Loaded {len(skills)} skills from {path.name}")
    return skills


# ---------- EDA on synthetic_dataset_extra.jsonl ----------

def eda_dataset(records):
    """
    records: list of dicts from synthetic_dataset_extra.jsonl
    Expected keys (best effort, handles missing):
    - job_description: str
    - skills: {skill_name: score} with score in {0, 0.5, 1}
    - bundle, domain, seniority (optional)
    - num_attempts, failure_reason (optional, for generation-quality)
    """

    n_examples = len(records)
    print("\n=== DATASET PROFILE ===")
    print(f"#examples: {n_examples}")

    # ----- job_description length -----
    lengths = []
    for r in records:
        text = r.get("job_description", "") or ""
        lengths.append(len(text.split()))

    if lengths:
        avg_len = sum(lengths) / len(lengths)
        print(f"Avg job description length: {avg_len:.1f} words")
        print(f"Min/Max job description length: {min(lengths)} / {max(lengths)} words")

        plt.figure(figsize=(6, 4))
        sns.histplot(lengths, bins=15, kde=False)
        plt.xlabel("Job description length (words)")
        plt.ylabel("#examples")
        plt.title("Distribution of job description length")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "job_description_length_hist.png")
        plt.close()
    else:
        print("[WARN] No job descriptions found.")

    # ----- skills per example -----
    n_explicit_per_ex = []
    n_implicit_per_ex = []

    all_skill_counts = Counter()

    bundles = []
    domains = []
    seniorities = []

    # stats per bundle based on dataset
    bundle_stats = {}  # bundle -> {"n": #examples, "exp": total_exp, "imp": total_imp}

    implicit_ratio = None
    n_unique_skills = 0

    for r in records:
        skills = r.get("skills") or {}
        exp = 0
        imp = 0
        for s, v in skills.items():
            try:
                v_float = float(v)
            except (ValueError, TypeError):
                continue
            if v_float >= 0.99:
                exp += 1
                all_skill_counts[s] += 1
            elif 0.25 <= v_float <= 0.75:
                imp += 1
                all_skill_counts[s] += 1

        n_explicit_per_ex.append(exp)
        n_implicit_per_ex.append(imp)

        bundle = r.get("bundle")
        if bundle:
            bundles.append(bundle)
            bs = bundle_stats.setdefault(bundle, {"n": 0, "exp": 0, "imp": 0})
            bs["n"] += 1
            bs["exp"] += exp
            bs["imp"] += imp

        domain = r.get("domain")
        if domain:
            domains.append(domain)

        seniority = r.get("seniority")
        if seniority:
            seniorities.append(seniority)

    if n_explicit_per_ex:
        avg_exp = sum(n_explicit_per_ex) / len(n_explicit_per_ex)
        avg_imp = sum(n_implicit_per_ex) / len(n_implicit_per_ex)
        print(f"Avg #skills per example: {avg_exp:.2f} explicit, {avg_imp:.2f} implicit")

        total_labels = sum(n_explicit_per_ex) + sum(n_implicit_per_ex)
        implicit_ratio = (sum(n_implicit_per_ex) / total_labels) if total_labels > 0 else 0.0
        print(f"Implicit skill ratio (over all labels): {implicit_ratio:.2%}")

        # histogram explicit / implicit counts
        plt.figure(figsize=(6, 4))
        sns.histplot(n_explicit_per_ex, color="C0", label="explicit", bins=10, kde=False)
        sns.histplot(n_implicit_per_ex, color="C1", label="implicit", bins=10, kde=False)
        plt.xlabel("#skills per example")
        plt.ylabel("#examples")
        plt.title("Distribution of explicit / implicit skills per example")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / "skills_per_example_hist.png")
        plt.close()

        # ----- pie: explicit vs implicit labels -----
        total_explicit = sum(n_explicit_per_ex)
        total_implicit = sum(n_implicit_per_ex)

        plt.figure(figsize=(4, 4))
        plt.pie(
            [total_explicit, total_implicit],
            labels=["explicit (1.0)", "implicit (0.5)"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4C72B0", "#DD8452"],
        )
        plt.title("Label distribution: explicit vs implicit")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "labels_explicit_vs_implicit_pie.png")
        plt.close()
    else:
        print("[WARN] No skill annotations found.")
        avg_exp = None
        avg_imp = None

    # ----- bundles / domains / seniority -----
    if bundles:
        bundle_counts = Counter(bundles)
        print("\nTop bundles (by #examples):")
        for b, c in bundle_counts.most_common(10):
            print(f"  {b}: {c}")

        # bar: #examples per bundle (Top-5)
        top_b = bundle_counts.most_common(5)
        labels = [b for b, _ in top_b]
        values = [c for _, c in top_b]

        plt.figure(figsize=(6, 4))
        sns.barplot(x=values, y=labels, orient="h")
        plt.xlabel("#examples")
        plt.ylabel("Bundle")
        plt.title("Examples per bundle (Top-5)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "bundles_top5_bar.png")
        plt.close()

        # bundle-level implicit ratio based on dataset
        bundle_summary = []
        for b, stats in bundle_stats.items():
            total_labels_b = stats["exp"] + stats["imp"]
            avg_imp_ratio_b = stats["imp"] / total_labels_b if total_labels_b > 0 else 0.0
            bundle_summary.append((b, stats["n"], avg_imp_ratio_b))

        bundle_summary.sort(key=lambda x: x[1], reverse=True)
        print("\nBundle summary from dataset:")
        print("bundle | #examples | avg_implicit_ratio")
        for b, n_ex, avg_imp_r in bundle_summary[:10]:
            print(f"{b} | {n_ex} | {avg_imp_r:.2f}")

        # bar: implicit ratio per bundle (Top-5)
        top_for_plot = bundle_summary[:5]
        if top_for_plot:
            labels = [row[0] for row in top_for_plot]
            avg_imp_ratios = [row[2] for row in top_for_plot]

            plt.figure(figsize=(7, 4))
            sns.barplot(x=avg_imp_ratios, y=labels, orient="h")
            plt.xlabel("Average implicit ratio (dataset)")
            plt.ylabel("Bundle")
            plt.xlim(0, 1)
            plt.title("Implicit skill ratio per bundle (Top-5)")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "bundles_implicit_ratio_top5_bar.png")
            plt.close()
    else:
        print("[INFO] No 'bundle' field found in dataset.")

    if domains:
        domain_counts = Counter(domains)
        print("\nDomains:")
        for d, c in domain_counts.most_common():
            print(f"  {d}: {c}")

    if seniorities:
        seniority_counts = Counter(seniorities)
        print("\nSeniorities:")
        for s, c in seniority_counts.most_common():
            print(f"  {s}: {c}")

    # ----- skills coverage -----
    print("\n=== SKILLS COVERAGE ===")
    n_unique_skills = len(all_skill_counts)
    print(f"#unique skills used in dataset: {n_unique_skills}")
    print("Top-10 skills by frequency:")
    for s, c in all_skill_counts.most_common(10):
        print(f"  {s}: {c}")

    skills_list = load_skills(SKILLS_PATH)
    if skills_list:
        all_known = set(skills_list)
        used = set(all_skill_counts.keys())
        unused = all_known - used
        print(f"\n#skills in global inventory: {len(all_known)}")
        print(f"#skills used at least once: {len(used)}")
        print(f"#skills never used: {len(unused)}")
        if unused:
            print("Example unused skills (up to 10):")
            for s in sorted(list(unused))[:10]:
                print(f"  {s}")

    if all_skill_counts:
        top_sk = all_skill_counts.most_common(15)
        labels = [s for s, _ in top_sk]
        values = [c for _, c in top_sk]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=values, y=labels, orient="h")
        plt.xlabel("Count")
        plt.ylabel("Skill")
        plt.title("Top-15 most frequent skills")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "skills_top15_bar.png")
        plt.close()

    return {
        "n_examples": n_examples,
        "avg_len": sum(lengths) / len(lengths) if lengths else None,
        "avg_exp": avg_exp,
        "avg_imp": avg_imp,
        "implicit_ratio": implicit_ratio,
        "n_unique_skills": n_unique_skills,
    }


# ---------- EDA on generation quality (if metadata exists) ----------

def eda_generation_quality(records):
    """
    Looks for fields:
    - num_attempts        (int)
    - failure_reason      (str or list)
    If they don't exist, prints a note and returns.
    """
    attempts = []
    reasons = []

    for r in records:
        if "num_attempts" in r:
            attempts.append(r.get("num_attempts"))
        if "failure_reason" in r and r["failure_reason"]:
            fr = r["failure_reason"]
            if isinstance(fr, str):
                reasons.append(fr)
            elif isinstance(fr, list):
                reasons.extend(fr)

    if not attempts and not reasons:
        print("\n=== GENERATION QUALITY ===")
        print("[INFO] No 'num_attempts' / 'failure_reason' metadata found in dataset.")
        print("       If you want this section, log these fields in generate_dataset.py.")
        return

    print("\n=== GENERATION QUALITY ===")

    if attempts:
        valid_attempts = [a for a in attempts if isinstance(a, (int, float))]
        if not valid_attempts:
            print("[WARN] 'num_attempts' present but not numeric.")
        else:
            avg_attempts = sum(valid_attempts) / len(valid_attempts)
            max_attempts = max(valid_attempts)
            print(f"Avg #generation attempts per example: {avg_attempts:.2f}")
            print(f"Max #generation attempts: {max_attempts}")
            n_with_retry = sum(1 for a in valid_attempts if a and a > 1)
            ratio_with_retry = n_with_retry / len(valid_attempts)
            print(f"Examples with ≥1 retry: {ratio_with_retry:.2%}")

            plt.figure(figsize=(6, 4))
            sns.histplot(
                valid_attempts,
                bins=range(1, int(max_attempts) + 2),
                discrete=True,
            )
            plt.xlabel("#generation attempts")
            plt.ylabel("#examples")
            plt.title("Distribution of generation attempts")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "generation_attempts_hist.png")
            plt.close()

    if reasons:
        main_tags = []
        for r in reasons:
            if not isinstance(r, str):
                continue
            tag = r.split(":", 1)[0]
            main_tags.append(tag)

        reason_counts = Counter(main_tags)
        print("Most common failure reasons:")
        for reason, c in reason_counts.most_common():
            print(f"  {reason}: {c}")

        labels = [r for r, _ in reason_counts.most_common()]
        values = [c for _, c in reason_counts.most_common()]
        plt.figure(figsize=(6, 4))
        sns.barplot(x=values, y=labels, orient="h")
        plt.xlabel("Count")
        plt.ylabel("Failure reason (tag)")
        plt.title("Generation failure reasons")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "generation_failure_reasons_bar.png")
        plt.close()


# ---------- (optional) EDA on plans_v1.jsonl ----------

def eda_plans(plans):
    """
    Optional EDA on the plan design (if you want it).
    Safe even if some fields (k, implicit_ratio_target, etc.) are missing.
    """
    if not plans:
        return

    print("\n=== PLANS DESIGN (OPTIONAL) ===")
    k_values = []
    n_exp_list = []
    n_imp_list = []
    bundle_counts = Counter()

    for p in plans:
        skills = p.get("skills") or {}
        k_values.append(len(skills))
        exp = sum(1 for _, v in skills.items() if isinstance(v, (int, float)) and v >= 0.99)
        imp = sum(1 for _, v in skills.items() if isinstance(v, (int, float)) and 0.25 <= v <= 0.75)
        n_exp_list.append(exp)
        n_imp_list.append(imp)

        bundle = p.get("bundle")
        if bundle:
            bundle_counts[bundle] += 1

    if k_values:
        avg_k = sum(k_values) / len(k_values)
        print(f"Avg #skills in plan: {avg_k:.2f}")
        print(f"Min/Max #skills in plan: {min(k_values)} / {max(k_values)}")

    if n_exp_list:
        avg_exp = sum(n_exp_list) / len(n_exp_list)
        avg_imp = sum(n_imp_list) / len(n_imp_list)
        print(f"Avg explicit/implicit per plan: {avg_exp:.2f} / {avg_imp:.2f}")

    if bundle_counts:
        print("Top bundles in plans:")
        for b, c in bundle_counts.most_common(10):
            print(f"  {b}: {c}")


# ---------- main ----------

def main():
    dataset = load_jsonl(DATASET_PATH)
    if not dataset:
        print("[ERROR] synthetic_dataset_extra.jsonl not found or empty. Aborting.")
        return

    _ = eda_dataset(dataset)
    eda_generation_quality(dataset)

    plans = load_jsonl(PLANS_PATH)
    if plans:
        eda_plans(plans)

    print("\n[INFO] EDA done.")
    print(f"[INFO] Plots saved under: {OUT_DIR}")
    print("\nTake from here the numbers for the single EDA slide:\n"
          " - #examples, #bundles, avg length, avg #skills, implicit ratio\n"
          " - Pie: labels_explicit_vs_implicit_pie.png\n"
          " - Bundles: bundles_top5_bar.png + bundles_implicit_ratio_top5_bar.png\n"
          " - Skills: skills_top15_bar.png\n"
          " - (Optional) generation quality & plans.")


if __name__ == "__main__":
    main()
