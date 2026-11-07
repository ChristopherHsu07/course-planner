"""Offline check of the parsers against HTML already saved under data/raw/.

Run from the scraper directory: ``python test_parsers.py``. It does not
download anything and does not touch the database. It reads the unscoped
cache (``form_page.html``, ``results_ASIA.html``, ``results_COMP.html``,
and ``detail_*.html`` sitting directly in ``data/raw/``), not the
term-scoped copies the crawler writes under ``data/raw/<term>/``.

The subject list is rewritten to ``data/subjects.json``. Each parsed
section is printed with its description cut to 60 characters.
"""

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

# Only the loose detail_*.html files, not data/raw/<term>/detail_*.html.
for path in sorted(RAW.glob("detail_*.html")):
    d = parse_detail(path.read_text())
    d["description"] = (d["description"] or "")[:60] + "..."
    print("\n", path.name)
    print(json.dumps(d, indent=1, default=str))
