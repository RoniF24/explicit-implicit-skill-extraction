from pathlib import Path

def load_skills(path: str) -> list[str]:
    skills: list[str] = []
    seen = set()

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s in seen:
            raise ValueError(f"Duplicate skill in skills file: {s}")
        seen.add(s)
        skills.append(s)

    if not skills:
        raise ValueError("Skills file is empty.")
    return skills

if __name__ == "__main__":
    base = Path(__file__).parent         
    skills = load_skills(str(base / "skills_v1.txt"))
    print(f"Loaded {len(skills)} skills:")
    for i, s in enumerate(skills, start=1):
        print(f"{i:03d}. {s}")