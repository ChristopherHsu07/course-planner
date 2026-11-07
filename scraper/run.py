"""Crawl one term of the Rice schedule and load it into Postgres.

The site rejects a search of the whole term, so the crawl walks one
subject at a time and then one CRN at a time:

1. Download the search form and save every subject to ``data/subjects.json``
   and to the ``subjects`` table.
2. For each wanted subject, download the results table and read its CRNs.
3. For each CRN, download the detail page, parse it, and upsert the section.
4. After a full crawl with no subject-level failures, mark sections that
   were not seen this run as inactive.

Each section is committed on its own, so one bad page does not roll back
the sections already stored. Failures and parser warnings go to
``data/crawl.log``. Cached HTML is reused unless ``--refresh`` is passed.

Example::

    python scraper/run.py
    python scraper/run.py --subjects ASIA,COMP --refresh
"""

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
# So `db.load` and the sibling scraper modules both import when this file
# is run as a script (python scraper/run.py) rather than as a package.
sys.path.insert(0, str(ROOT))

from fetch import get                                        # noqa: E402
from parse_detail import parse_detail                        # noqa: E402
from parse_schedule import parse_results, parse_subjects     # noqa: E402
from db.load import (connect, mark_inactive, upsert_section,  # noqa: E402
                     upsert_subjects)

LOG = ROOT / "data" / "crawl.log"


def log(msg: str):
    """Append one line to ``data/crawl.log``.

    Used for subject failures, section failures, and parser warnings.
    Progress that a person is watching is printed, not written here.
    """
    with LOG.open("a") as f:
        f.write(msg + "\n")


def main():
    """Run the crawl described in the module docstring.

    ``--term`` defaults to ``202720`` (Spring 2027). ``--subjects`` limits
    the crawl to a comma-separated list of codes and also disables inactive
    marking, because a partial crawl cannot tell which other sections
    disappeared. ``--refresh`` re-downloads every page instead of reading
    the HTML cache.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="202720")
    ap.add_argument("--subjects", help="comma-separated codes, e.g. ASIA,COMP")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cache and re-download everything")
    args = ap.parse_args()
    term = args.term

    # The form is fetched even for a partial crawl: it is the subject catalog.
    form_html = get({"p_action": "QUERY", "p_term": term, "p_mode": "AND"},
                    f"{term}/form_page", force=args.refresh)
    all_subjects = parse_subjects(form_html)
    (ROOT / "data" / "subjects.json").write_text(
        json.dumps(all_subjects, indent=2))

    wanted = all_subjects
    if args.subjects:
        codes = {c.strip().upper() for c in args.subjects.split(",")}
        wanted = [s for s in all_subjects if s["code"] in codes]
        missing = codes - {s["code"] for s in wanted}
        if missing:
            sys.exit(f"unknown subject codes: {sorted(missing)}")

    # Inactive marking is only valid when every subject in the term was crawled.
    full_crawl = args.subjects is None
    complete = True              # becomes False if any subject page fails
    keep, loaded, failed = [], 0, 0
    start = time.time()

    with connect() as conn:
        # Always refresh the full catalog, even when only some subjects are crawled.
        upsert_subjects(conn, all_subjects)
        conn.commit()

        for si, subj in enumerate(wanted, 1):
            code = subj["code"]
            try:
                html = get({"p_action": "QUERY", "p_term": term,
                            "p_mode": "AND", "p_subj": code},
                           f"{term}/results_{code}", force=args.refresh)
                rows = parse_results(html)
            except Exception as e:
                # A missed subject must not look like "those sections were cancelled".
                complete = False
                log(f"SUBJECT FAIL {code}: {e!r}")
                print(f"[{si}/{len(wanted)}] {code}: FAILED ({e!r})")
                continue

            print(f"[{si}/{len(wanted)}] {code}: {len(rows)} sections "
                  f"({(time.time() - start) / 60:.1f} min elapsed)")

            for row in rows:
                crn = row["crn"]
                # Record the CRN before the detail fetch. A section that fails
                # to parse still counts as seen, so it is not marked inactive.
                keep.append(crn)
                try:
                    dhtml = get({"p_action": "COURSE", "p_term": term,
                                 "p_crn": crn},
                                f"{term}/detail_{crn}", force=args.refresh)
                    d = parse_detail(dhtml)
                    # Guard against a cached page from a different term.
                    if d["term_code"] != term:
                        raise ValueError(f"term mismatch: {d['term_code']}")
                    with conn.cursor() as cur:
                        upsert_section(cur, d)
                    conn.commit()
                    loaded += 1
                    for w in d["warnings"]:
                        log(f"WARN {term} {crn} {d['subject_code']} "
                            f"{d['course_number']}: {w}")
                except Exception as e:
                    conn.rollback()
                    failed += 1
                    log(f"SECTION FAIL {term} {crn} ({row['course']}): {e!r}")

        if full_crawl and complete:
            n = mark_inactive(conn, term, keep)
            conn.commit()
            print(f"marked {n} sections inactive")
        elif full_crawl:
            print("skipped inactive-marking because some subjects failed")

    print(f"\ndone: {loaded} loaded, {failed} failed, "
          f"{(time.time() - start) / 60:.1f} min. See data/crawl.log")


if __name__ == "__main__":
    main()
