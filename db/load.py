"""Write parsed schedule data into Postgres.

``connect`` opens the database named by ``DATABASE_URL``. ``upsert_subjects``
refreshes the subject catalog. ``upsert_section`` writes one section and
replaces its meetings and instructor links. ``mark_inactive`` flags sections
that a finished full crawl no longer saw.

The caller owns transactions: these functions execute and return, and do
not commit. ``embedding`` is intentionally absent from the section write,
so a later re-upsert leaves any stored vector in place.
"""

import os

import psycopg

# Columns copied from a parse_detail() dict into the sections table.
# term_code and crn are the primary key and are not overwritten on conflict.
# scraped_at and is_active are set in the SQL, not taken from the dict.
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
    """Open a connection using ``DATABASE_URL``.

    Example: ``postgresql://planner:planner@localhost:5433/rice_planner``.
    The caller should use the connection as a context manager so it closes.
    """
    return psycopg.connect(os.environ["DATABASE_URL"])


def upsert_subjects(conn, subjects: list[dict]):
    """Insert each ``{"code", "name"}`` subject, updating the name on conflict.

    Does not commit. A subject that disappears from a later form page is
    left in the table; only the name of a known code is refreshed.
    """
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO subjects (code, name) VALUES (%s, %s) "
            "ON CONFLICT (code) DO UPDATE SET name = EXCLUDED.name",
            [(s["code"], s["name"]) for s in subjects],
        )


def upsert_section(cur, d: dict):
    """Insert or update one parsed section, then replace its child rows.

    ``d`` is the dict returned by ``parse_detail``. The section row is
    matched on ``(term_code, crn)``. A repeat crawl overwrites every other
    column, sets ``scraped_at`` to now, and marks the section active again.

    Meetings and ``section_instructors`` rows for this section are deleted
    and inserted again. Replacing them wholesale means a schedule change
    cannot leave a stale weekday or instructor behind. Instructor people
    are upserted by key (netid, or ``name:<name>`` when there is no netid);
    the link row is inserted only if that section does not already point
    at that instructor.

    Uses the cursor it is given and does not commit or roll back.
    """
    d = dict(d)
    # Collapse whitespace, including non-breaking spaces, to single spaces.
    d["meeting_raw"] = " ".join((d["meeting_raw"] or "").split())

    cols = ", ".join(SECTION_COLS)
    vals = ", ".join(f"%({c})s" for c in SECTION_COLS)
    # The primary key is the conflict target, so it is not assigned again.
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
    """Flag sections of ``term`` whose CRN is not in ``keep_crns``.

    Called only after a full crawl in which every subject page succeeded.
    Rows stay in the table; ``is_active`` becomes false so a section that
    left the schedule is distinguishable from one that is still offered.
    Returns the number of rows updated. Does not commit.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE sections SET is_active = FALSE "
            "WHERE term_code = %s AND NOT (crn = ANY(%s))",
            (term, keep_crns),
        )
        return cur.rowcount
