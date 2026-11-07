import os

import psycopg

SECTION_COLS = [
    "term_code", "crn", "subject_code", "course_number", "section_no",
    "title", "long_title", "description", "department", "part_of_term",
    "start_date", "end_date", "credits", "course_type", "grade_mode",
    "language", "distribution_group", "method", "restrictions",
    "max_enrollment", "enrollment_note", "permission_required", "enrolled",
    "waitlisted", "waitlist_max", "enrollment_as_of", "final_exam",
    "meeting_raw", "analyzing_diversity",
    "crosslist_max_enrollment", "crosslist_enrolled",
]


def connect():
    # e.g. postgresql://planner:planner@localhost:5433/rice_planner
    return psycopg.connect(os.environ["DATABASE_URL"])


def upsert_subjects(conn, subjects: list[dict]):
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO subjects (code, name) VALUES (%s, %s) "
            "ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name",
            [(s["code"], s["name"]) for s in subjects],
        )


def upsert_section(cur, d: dict):
    """Insert or update one parsed section with its meetings and instructors.
    Meetings and instructor links are replaced wholesale, so a changed
    schedule never leaves stale rows behind."""
    d = dict(d)
    d["meeting_raw"] = " ".join((d["meeting_raw"] or "").split())  # fix nbsp

    cols = ", ".join(SECTION_COLS)
    vals = ", ".join(f"%({c})s" for c in SECTION_COLS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in SECTION_COLS
                        if c not in ("term_code", "crn"))
    cur.execute(
        f"INSERT INTO sections ({cols}, scraped_at, is_active) "
        f"VALUES ({vals}, now(), TRUE) "
        f"ON CONFLICT (term_code, crn) DO UPDATE SET {updates}, "
        f"scraped_at = now(), is_active = TRUE",
        d,
    )

    key = (d["term_code"], d["crn"])

    cur.execute("DELETE FROM meetings WHERE term_code = %s AND crn = %s", key)
    for m in d["meetings"]:
        cur.execute(
            "INSERT INTO meetings (term_code, crn, day_of_week, start_time, "
            "end_time, start_date, end_date) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (*key, m["day_of_week"], m["start_time"], m["end_time"],
             m["start_date"], m["end_date"]),
        )

    cur.execute(
        "DELETE FROM section_instructors WHERE term_code = %s AND crn = %s",
        key)
    for ins in d["instructors"]:
        cur.execute(
            "INSERT INTO instructors (key, netid, name) VALUES (%s,%s,%s) "
            "ON CONFLICT (key) DO UPDATE SET netid = EXCLUDED.netid, "
            "name = EXCLUDED.name",
            (ins["key"], ins["netid"], ins["name"]),
        )
        cur.execute(
            "INSERT INTO section_instructors (term_code, crn, instructor_key) "
            "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
            (*key, ins["key"]),
        )


def mark_inactive(conn, term: str, keep_crns: list[str]):
    """After a full crawl, flag sections that no longer appear."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sections SET is_active = FALSE "
            "WHERE term_code = %s AND NOT (crn = ANY(%s))",
            (term, keep_crns),
        )
        return cur.rowcount