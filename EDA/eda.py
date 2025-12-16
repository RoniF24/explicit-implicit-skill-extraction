from pathlib import Path
import sys
import json
import hashlib
from collections import Counter
import statistics as stats
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import unicodedata
from itertools import combinations
import re

# CONFIG AND PATHS
ROOT_DIR= Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT_DIR))
JSONL_PATH = ROOT_DIR / "data" / "synthetic_dataset.jsonl"

# LOAD GLOBAL SKILL VECTOR
from skills.globalVector import GLOBAL_SKILL_VECTOR

global_set = set(GLOBAL_SKILL_VECTOR)

# ACCUMULATORS (counters & feature lists)
loaded = 0
missing_required = 0
bad_skills_type = 0
unknown_skills = 0
bad_label_values = 0
seen_hashes = set()
duplicate_count = 0
text_len_chars = []
text_len_words = []
explicit_per_sample = []
implicit_per_sample = []
# per-skill counts (explicit/implicit)
explicit_counts = Counter()
implicit_counts = Counter()
#co-occurrence counter (pairs of skills)
pair_counts = Counter()
#LENGTH (sentence count) — target is 4–6 sentences
sentence_counts = []



#Text Cleanliness Checks (Whitespace & Weird Characters)
double_space_texts = 0
non_ascii_texts = 0
control_char_texts = 0

# SINGLE PASS OVER DATASET (Sanity + Length + Duplicates + Cleanliness)
with open(JSONL_PATH, "r", encoding="utf-8") as f:

    for line_num, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        obj = json.loads(line)
        loaded += 1
    # Sanity checks: required fields + skills format
        if "job_description" not in obj or "skills" not in obj:
            missing_required += 1
            continue
        
        text = obj["job_description"]

        skills = obj["skills"]
        sample_explicit = 0
        sample_implicit = 0

        if not isinstance(skills, dict):
            bad_skills_type += 1
            continue

        #count skill pairs per sample (co-occurrence)
        skill_list = sorted(skills.keys())
        for a, b in combinations(skill_list, 2):
            pair_counts[(a, b)] += 1


        # Sanity checks: skill names & label values 
        for skill, label in skills.items():
            if skill not in global_set:
                unknown_skills += 1
            if label not in (0.5, 1.0):
                bad_label_values += 1
            if label == 1.0:
                sample_explicit += 1
            elif label == 0.5:
                sample_implicit += 1
            # count labels per skill
            if label == 1.0:
                explicit_counts[skill] += 1
            elif label == 0.5:
                implicit_counts[skill] += 1

        explicit_per_sample.append(sample_explicit)
        implicit_per_sample.append(sample_implicit)

                
        # Duplicate detection (exact text hash) 
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()

        if h in seen_hashes:
            duplicate_count += 1
        else:
            seen_hashes.add(h)

        text_len_chars.append(len(text))
        text_len_words.append(len(text.split()))

        #DOUBLE SPACES
        if "  " in text:
            double_space_texts += 1
        #NON-ASCII CHARACTERS
        if any(ord(ch) > 127 for ch in text):
            non_ascii_texts += 1

        #CONTROL/BREACK CHARACTERS
        has_control = False
        for ch in text:
            if unicodedata.category(ch) == "Cc" and ch not in ("\n", "\t", "\r"):
                has_control = True
                break
        if has_control:
            control_char_texts += 1

        # rough sentence split by . ! ?
        parts = [p for p in re.split(r"[.!?]+", text) if p.strip()]
        sentence_counts.append(len(parts))

        

# SANITY REPORT + TEXT LENGTH SUMMARY
print("Loaded records:", loaded)
print("Missing required keys:", missing_required) 
print("Bad skills type:", bad_skills_type)
print("Unknown skills:", unknown_skills)
print("Bad label values:", bad_label_values)
print("Exact duplicate job_descriptions:", duplicate_count)
print("Unique job_descriptions:", len(seen_hashes))
print("=== TEXT LENGTH STATS ===")
print("Samples:", len(text_len_words))
print("\nChars:")
print("  mean:", round(stats.mean(text_len_chars), 2))
print("  median:", stats.median(text_len_chars))
print("  min/max:", min(text_len_chars), "/", max(text_len_chars))
print("\nWords:")
print("  mean:", round(stats.mean(text_len_words), 2))
print("  median:", stats.median(text_len_words))
print("  min/max:", min(text_len_words), "/", max(text_len_words))

#OUTPUTS
OUTPUT_DIR = ROOT_DIR / "EDA"
OUTPUT_DIR.mkdir(exist_ok=True)

#HISTOGRAMA: WORDS
plt.figure()
plt.hist(text_len_words, bins=20)
plt.title("Job Description Length (Words)")
plt.xlabel("Words")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_words.png", dpi=200)
plt.show()

#HISTOGRAMA: CHARS
plt.figure()
plt.hist(text_len_chars, bins=20)
plt.title("Job Description Length (Chars)")
plt.xlabel("Characters")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_chars.png", dpi=200)
plt.show()
print("Saved plots to:", OUTPUT_DIR)

#TEXT CLEANLINESS REPORT
print("\n Text Cleanliness")
print("Texts with double spaces:", double_space_texts)
print("Texts with non-ASCII chars:", non_ascii_texts)
print("Texts with control chars:", control_char_texts)

# sanity: total label counts
print("\nSkill label counts (sanity)")
print("Total explicit labels:", sum(explicit_counts.values()))
print("Total implicit labels:", sum(implicit_counts.values()))

print("\nper-sample label stats ")
print("Avg explicit per sample:", round(stats.mean(explicit_per_sample), 2))
print("Avg implicit per sample:", round(stats.mean(implicit_per_sample), 2))
print("Samples with 0 implicit:", sum(1 for x in implicit_per_sample if x == 0))

TOP_K = 20

# Top skills (explicit / implicit)
print("\n Top skills by label")
print(f"\nTop {TOP_K} EXPLICIT (1.0):")
for skill, cnt in explicit_counts.most_common(TOP_K):
    print(f"  {skill}: {cnt}")

print(f"\nTop {TOP_K} IMPLICIT (0.5):")
for skill, cnt in implicit_counts.most_common(TOP_K):
    print(f"  {skill}: {cnt}")

#Bottom skills (total occurrences, including zeros)
print(f"\nBottom {TOP_K} TOTAL (lowest occurrences):")
total_counts = Counter()
for s in GLOBAL_SKILL_VECTOR:
    total_counts[s] = explicit_counts.get(s, 0) + implicit_counts.get(s, 0)
for skill, cnt in sorted(total_counts.items(), key=lambda x: x[1])[:TOP_K]:
    print(f"  {skill}: {cnt}")

# imbalance summary
print("\nImbalance summary")

# 1) zero-occurrence skills
zero_skills = [s for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) == 0]
print("Skills with 0 occurrences:", len(zero_skills))

# 2) rare skills thresholds
lt10 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 10)
lt20 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 20)
print("Skills with <10 occurrences:", lt10)
print("Skills with <20 occurrences:", lt20)

# 3) dominance: share covered by top skills
total_labels = sum(total_counts.values())
top10_sum = sum(cnt for _, cnt in total_counts.most_common(10))
top20_sum = sum(cnt for _, cnt in total_counts.most_common(20))

print("Total labeled skill occurrences:", total_labels)
print("Top 10 skills cover:", round(100 * top10_sum / total_labels, 2), "%")
print("Top 20 skills cover:", round(100 * top20_sum / total_labels, 2), "%")

TOP_PAIRS = 10

#Top co-occurring skill pairs
print(f"\nTop {TOP_PAIRS} co-occurring skill pairs")
for (a, b), cnt in pair_counts.most_common(TOP_PAIRS):
    print(f"  ({a}, {b}): {cnt}")

#Length — sentence-count distribution and % within target (4–6 sentences)
print("\nLength (Sentence Count)")
print("Sentences - mean:", round(stats.mean(sentence_counts), 2))
print("Sentences - median:", stats.median(sentence_counts))
print("Sentences - min/max:", min(sentence_counts), "/", max(sentence_counts))

in_4_6 = sum(1 for x in sentence_counts if 4 <= x <= 6)
too_short = sum(1 for x in sentence_counts if x < 4)
too_long = sum(1 for x in sentence_counts if x > 6)

print("Samples with 4–6 sentences:", in_4_6, f"({in_4_6/len(sentence_counts)*100:.1f}%)")
print("Too short (<4):", too_short, f"({too_short/len(sentence_counts)*100:.1f}%)")
print("Too long (>6):", too_long, f"({too_long/len(sentence_counts)*100:.1f}%)")


#Types — error rates (%)
total = loaded if loaded else 1

print("\nTypes (Rates)")
print("Missing required keys (%):", round(100 * missing_required / total, 2))
print("Bad skills type (%):", round(100 * bad_skills_type / total, 2))
print("Duplicates (%):", round(100 * duplicate_count / total, 2))
print("Non-ASCII (%):", round(100 * non_ascii_texts / total, 2))
print("Double spaces (%):", round(100 * double_space_texts / total, 2))
print("Control chars (%):", round(100 * control_char_texts / total, 2))
print("Unknown skill entries (count):", unknown_skills)
print("Bad label entries (count):", bad_label_values)

# SUMMARY WITH PERCENTAGES
total_samples = loaded if loaded else 1
total_labels = (sum(explicit_counts.values()) + sum(implicit_counts.values())) or 1

print("\n=== Summary (Percentages) ===")
print("Missing required keys (% samples):", round(100 * missing_required / total_samples, 2))
print("Bad skills type (% samples):", round(100 * bad_skills_type / total_samples, 2))
print("Duplicates (% samples):", round(100 * duplicate_count / total_samples, 2))

print("Non-ASCII (% samples):", round(100 * non_ascii_texts / total_samples, 2))
print("Double spaces (% samples):", round(100 * double_space_texts / total_samples, 2))
print("Control chars (% samples):", round(100 * control_char_texts / total_samples, 2))

print("Explicit labels (% of labeled skills):", round(100 * sum(explicit_counts.values()) / total_labels, 2))
print("Implicit labels (% of labeled skills):", round(100 * sum(implicit_counts.values()) / total_labels, 2))

print("Samples with 0 implicit (% samples):", round(100 * (sum(1 for x in implicit_per_sample if x == 0)) / total_samples, 2))

#Numbers for dashboard
total_samples = loaded if loaded else 1
total_labels = sum(explicit_counts.values()) + sum(implicit_counts.values())
total_labels = total_labels if total_labels else 1

explicit_pct = 100 * sum(explicit_counts.values()) / total_labels
implicit_pct = 100 * sum(implicit_counts.values()) / total_labels
zero_implicit_pct = 100 * (sum(1 for x in implicit_per_sample if x == 0) / total_samples)

non_ascii_pct = 100 * non_ascii_texts / total_samples
dup_pct = 100 * duplicate_count / total_samples

# Top skills (keep small to avoid clutter)
TOP_K = 5
top_exp = explicit_counts.most_common(TOP_K)
top_imp = implicit_counts.most_common(TOP_K)

# Imbalance summary (you already computed total_counts in A2/A4)
# If total_counts doesn't exist here, recreate it:
try:
    _ = total_counts
except NameError:
    total_counts = Counter()
    for s in GLOBAL_SKILL_VECTOR:
        total_counts[s] = explicit_counts.get(s, 0) + implicit_counts.get(s, 0)

zero_skills = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) == 0)
lt10 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 10)
lt20 = sum(1 for s in GLOBAL_SKILL_VECTOR if total_counts.get(s, 0) < 20)
top10_cover = 100 * (sum(cnt for _, cnt in total_counts.most_common(10)) / sum(total_counts.values()))

# Length summary (sentence constraint)
sent_mean = stats.mean(sentence_counts)
sent_median = stats.median(sentence_counts)
sent_min, sent_max = min(sentence_counts), max(sentence_counts)

words_mean = stats.mean(text_len_words)
words_median = stats.median(text_len_words)
words_min, words_max = min(text_len_words), max(text_len_words)

chars_mean = stats.mean(text_len_chars)
chars_median = stats.median(text_len_chars)
chars_min, chars_max = min(text_len_chars), max(text_len_chars)

# --- Build 2x2 dashboard ---
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.3)

# (1) KPI boxes (top-left)
ax0 = fig.add_subplot(gs[0, 0])
ax0.axis("off")
ax0.text(0.0, 1.0, "EDA Summary", fontsize=16, fontweight="bold", va="top")
ax0.text(0.0, 0.78, f"Samples: {total_samples}", fontsize=14)
ax0.text(0.0, 0.62, f"Global skills: {len(GLOBAL_SKILL_VECTOR)}", fontsize=14)
ax0.text(0.0, 0.46, f"Total labeled skill occurrences: {sum(total_counts.values())}", fontsize=14)
ax0.text(0.0, 0.25, f"Explicit vs Implicit: {explicit_pct:.1f}% / {implicit_pct:.1f}%", fontsize=12)
ax0.text(0.0, 0.12, f"Samples with 0 implicit: {zero_implicit_pct:.1f}%", fontsize=12)

# (2) Donut: explicit vs implicit (top-right)
ax1 = fig.add_subplot(gs[0, 1])
vals = [sum(explicit_counts.values()), sum(implicit_counts.values())]
labels = ["Explicit (1.0)", "Implicit (0.5)"]
wedges, _ = ax1.pie(vals, startangle=90, wedgeprops=dict(width=0.45))
ax1.set_title("Label Share", fontsize=12)
ax1.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5))

# (3) Top skills bar charts (bottom-left)
ax2 = fig.add_subplot(gs[1, 0])
# build combined view: explicit top + implicit top
names = [s for s, _ in top_exp] + [s for s, _ in top_imp]
counts = [c for _, c in top_exp] + [c for _, c in top_imp]
colors = ["tab:blue"] * len(top_exp) + ["tab:orange"] * len(top_imp)

ax2.barh(range(len(names)), counts, color=colors)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names)
ax2.invert_yaxis()
ax2.set_title("Top 5 Explicit (blue) + Top 5 Implicit (orange)", fontsize=12)
ax2.set_xlabel("Count")

# (4) Quality + Length table-like text (bottom-right)
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis("off")
ax3.set_title("Quality (Types) + Length", fontsize=12, fontweight="bold", loc="left")

quality_lines = [
    f"Missing keys: {missing_required} ({100*missing_required/total_samples:.1f}%)",
    f"Bad skills type: {bad_skills_type} ({100*bad_skills_type/total_samples:.1f}%)",
    f"Duplicates: {duplicate_count} ({dup_pct:.1f}%)",
    f"Non-ASCII: {non_ascii_texts} ({non_ascii_pct:.1f}%)",
    f"Double spaces: {double_space_texts} ({100*double_space_texts/total_samples:.1f}%)",
    f"Control chars: {control_char_texts} ({100*control_char_texts/total_samples:.1f}%)",
    f"Unknown skills: {unknown_skills}",
    f"Bad label values: {bad_label_values}",
    "",
    f"Sentence count: mean {sent_mean:.2f}, median {sent_median:.1f}, min/max {sent_min}/{sent_max}",
    f"Words: mean {words_mean:.1f}, median {words_median:.1f}, min/max {words_min}/{words_max}",
    f"Chars: mean {chars_mean:.1f}, median {chars_median:.1f}, min/max {chars_min}/{chars_max}",
    "",
    f"Imbalance: zero-occ {zero_skills}, <10 {lt10}, <20 {lt20}, Top10 cover {top10_cover:.2f}%",
]

ax3.text(0.0, 1.0, "\n".join(quality_lines), va="top", fontsize=11)

# Footer note
fig.text(0.01, 0.01, "Co-occurrence: Top pair appears 10/1000 (no dominant templates).", fontsize=10)

out_path = OUTPUT_DIR / "eda_dashboard.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200)
plt.close(fig)

print("Saved dashboard to:", out_path)

