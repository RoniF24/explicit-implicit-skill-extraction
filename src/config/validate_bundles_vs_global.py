import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]  # src/
BUNDLES_PATH = ROOT / "config" / "bundles_v1.json"
SKILLS_PATH  = ROOT / "skills" / "skills_v1.txt"

def load_skills_txt(path: Path) -> list[str]:
    skills = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s in seen:
            raise ValueError(f"Duplicate skill in skills_v1.txt: {s}")
        seen.add(s)
        skills.append(s)
    return skills

def bundle_skill_set(b: dict) -> set[str]:
    s = set(b.get("must_have", [])) | set(b.get("optional", []))
    for group in b.get("pick_one_of", []):
        s |= set(group)
    return s

def main():
    global_skills = load_skills_txt(SKILLS_PATH)
    global_set = set(global_skills)

    data = json.loads(BUNDLES_PATH.read_text(encoding="utf-8"))
    bundles = data.get("bundles", [])

    # 1) id uniqueness
    ids = [b.get("id") for b in bundles]
    dup_ids = [i for i,c in Counter(ids).items() if c > 1]
    if dup_ids:
        print("❌ Duplicate bundle ids:", dup_ids)

    # 2) unknown skills in bundles
    used = set()
    unknown = []
    for b in bundles:
        sset = bundle_skill_set(b)
        used |= sset
        for s in sorted(sset):
            if s not in global_set:
                unknown.append((b.get("id"), s))

    if unknown:
        print("\n❌ Skills used in bundles but NOT in skills_v1.txt:")
        for bid, s in unknown:
            print(f"  - {bid}: {s}")
    else:
        print("✅ All bundle skills exist in skills_v1.txt")

    # 3) dead skills in global
    dead = sorted(global_set - used)
    print(f"\nUsed skills in bundles: {len(used)}")
    print(f"Global skills: {len(global_set)}")
    print(f"Dead skills (in global but not used in any bundle): {len(dead)}")
    for s in dead:
        print("  -", s)

if __name__ == "__main__":
    main()
