import json
from fetch import RAW
from parse_detail import parse_detail
from parse_schedule import parse_results, parse_subjects

subjects = parse_subjects((RAW / "form_page.html").read_text())
(RAW.parent / "subjects.json").write_text(json.dumps(subjects, indent=2))
print(len(subjects), "subjects, e.g.", subjects[:3])

for subj in ("ASIA", "COMP"):
    rows = parse_results((RAW / f"results_{subj}.html").read_text())
    print(f"{subj}: {len(rows)} sections;", rows[:2])

for path in sorted(RAW.glob("detail_*.html")):
    d = parse_detail(path.read_text())
    d["description"] = (d["description"] or "")[:60] + "..."
    print("\n", path.name)
    print(json.dumps(d, indent=1, default=str))