"""Section search and lookup: ``search_sections``, ``get_section``,
``get_course_sections`` and ``seat_status``.

Also holds the pieces the scheduling module reuses: ``build_filter_clauses``
turns a ``SectionFilters`` into SQL conditions, and ``query_summaries`` runs
a section query and returns ``SectionSummary`` rows.

All functions take an open psycopg connection first and only read.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable, Optional, Sequence

from .db import cursor, resolve_term
from .models import (DAY_LETTERS, Instructor, Meeting, SeatStatus,
                     SectionDetail, SectionFilters, SectionSummary, hhmm)

MAX_LIMIT = 200

# Short forms of the distribution group names accepted by the filter.
_GROUP_SHORT = {"I", "II", "III"}

# "COMP 140", "comp140", "COMP-140", "MATH 101L": a whole-query course code.
_CODE_RE = re.compile(r"^\s*([A-Za-z]{2,5})\s*-?\s*(\d{3,4}[A-Za-z]?)\s*$")

# Smallest and largest number in a credits string such as '3', '1 TO 4'
# or '1 OR 3'. 'X OR Y' is treated as the range X..Y, a slight over-match.
_CREDITS_MIN = ("(SELECT min(r.m[1]::numeric) FROM "
                "regexp_matches(s.credits, '[0-9]+(?:\\.[0-9]+)?', 'g') AS r(m))")
_CREDITS_MAX = _CREDITS_MIN.replace("min(", "max(")

# Leading digits of the course number, as an int (NULL if none).
_LEVEL = "NULLIF(substring(s.course_number from '^[0-9]+'), '')::int"


def _like(text: str) -> str:
    """Escape ``%``, ``_`` and ``\\`` so ``text`` matches literally in LIKE."""
    return re.sub(r"([\\%_])", r"\\\1", text)


def _clamp_limit(limit: int) -> int:
    return max(1, min(int(limit), MAX_LIMIT))


def build_filter_clauses(filters: Optional[SectionFilters],
                         term: str) -> tuple[list[str], dict[str, Any]]:
    """Translate ``filters`` into SQL conditions on ``sections s``.

    Returns ``(clauses, params)`` where clauses are ANDed and params use
    ``%(name)s`` placeholders. Always restricts to ``term`` and to active
    sections, and hides grouped Research sections unless asked.
    """
    f = filters or SectionFilters()
    clauses = ["s.term_code = %(term)s", "s.is_active"]
    params: dict[str, Any] = {"term": term}

    if not f.include_grouped:
        clauses.append("s.group_id IS NULL")

    if f.subject:
        subjects = [f.subject] if isinstance(f.subject, str) else list(f.subject)
        params["subjects"] = [x.strip().upper() for x in subjects]
        clauses.append("s.subject_code = ANY(%(subjects)s)")

    if f.distribution_group:
        group = f.distribution_group.strip()
        if group.upper() in _GROUP_SHORT:
            group = f"Distribution Group {group.upper()}"
        params["dist"] = group
        clauses.append("lower(s.distribution_group) = lower(%(dist)s)")

    if f.method:
        params["method"] = f.method.strip()
        clauses.append("lower(s.method) = lower(%(method)s)")

    if f.level_min is not None:
        params["level_min"] = f.level_min
        clauses.append(f"{_LEVEL} >= %(level_min)s")
    if f.level_max is not None:
        params["level_max"] = f.level_max
        clauses.append(f"{_LEVEL} <= %(level_max)s")

    if f.instructor and f.instructor.strip():
        params["instr"] = f.instructor.strip()
        params["instr_like"] = f"%{_like(f.instructor.strip())}%"
        clauses.append(
            "EXISTS (SELECT 1 FROM section_instructors si "
            "JOIN instructors i ON i.key = si.instructor_key "
            "WHERE si.term_code = s.term_code AND si.crn = s.crn "
            "AND (lower(i.netid) = lower(%(instr)s) "
            "OR i.name ILIKE %(instr_like)s))")

    if f.open_seats_only:
        clauses.append("s.max_enrollment - s.enrolled > 0")

    # A section matches if its allowed credit range overlaps the request.
    if f.credits_min is not None:
        params["credits_min"] = f.credits_min
        clauses.append(f"{_CREDITS_MAX} >= %(credits_min)s")
    if f.credits_max is not None:
        params["credits_max"] = f.credits_max
        clauses.append(f"{_CREDITS_MIN} <= %(credits_max)s")

    return clauses, params


def format_meeting_summary(meetings: Iterable[tuple[int, str, str]]) -> str:
    """Compact text for ``(day, "HH:MM", "HH:MM")`` tuples.

    Days sharing a time are joined: ``"MW 14:00-15:15; W 16:00-17:15"``.
    """
    by_time: dict[tuple[str, str], set[int]] = defaultdict(set)
    for day, start, end in meetings:
        by_time[(start, end)].add(day)
    parts = []
    for (start, end), days in sorted(
            by_time.items(), key=lambda kv: (min(kv[1]), kv[0])):
        letters = "".join(DAY_LETTERS[d] for d in sorted(days))
        parts.append(f"{letters} {start}-{end}")
    return "; ".join(parts)


def _open_seats(max_enrollment: Optional[int],
                enrolled: Optional[int]) -> Optional[int]:
    if max_enrollment is None or enrolled is None:
        return None
    return max(0, max_enrollment - enrolled)


def query_summaries(conn, clauses: Sequence[str], params: dict[str, Any], *,
                    order_by: str, limit: int,
                    score_sql: str = "NULL::float8",
                    extra_params: Optional[dict[str, Any]] = None
                    ) -> list[SectionSummary]:
    """Run a section query and build ``SectionSummary`` rows.

    ``clauses`` are ANDed into WHERE (they may reference ``s``); ``score_sql``
    is an expression selected as ``score``; ``order_by`` must be trusted SQL
    (it is never built from user text). Instructors and meetings are loaded
    for the returned rows only.
    """
    sql = f"""
        SELECT s.term_code, s.crn, s.subject_code, s.course_number,
               s.section_no, s.title, s.credits, s.distribution_group,
               s.method, s.course_type, s.max_enrollment, s.enrolled,
               {score_sql} AS score
        FROM sections s
        WHERE {' AND '.join(clauses)}
        ORDER BY {order_by}
        LIMIT %(limit)s
    """
    p = {**params, **(extra_params or {}), "limit": _clamp_limit(limit)}
    with cursor(conn) as cur:
        cur.execute(sql, p)
        rows = cur.fetchall()
    if not rows:
        return []

    term = rows[0]["term_code"]
    crns = [r["crn"] for r in rows]
    names = _instructor_names(conn, term, crns)
    meetings = _meeting_tuples(conn, term, crns)
    return [
        SectionSummary(
            term_code=r["term_code"], crn=r["crn"],
            subject_code=r["subject_code"], course_number=r["course_number"],
            section_no=r["section_no"], title=r["title"],
            credits=r["credits"], distribution_group=r["distribution_group"],
            method=r["method"], course_type=r["course_type"],
            instructors=names.get(r["crn"], []),
            meeting_summary=format_meeting_summary(meetings.get(r["crn"], [])),
            max_enrollment=r["max_enrollment"], enrolled=r["enrolled"],
            open_seats=_open_seats(r["max_enrollment"], r["enrolled"]),
            score=float(r["score"]) if r["score"] is not None else None,
        )
        for r in rows
    ]


def _instructor_names(conn, term: str, crns: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    with cursor(conn) as cur:
        cur.execute(
            "SELECT si.crn, i.name FROM section_instructors si "
            "JOIN instructors i ON i.key = si.instructor_key "
            "WHERE si.term_code = %s AND si.crn = ANY(%s) ORDER BY i.name",
            (term, crns))
        for r in cur.fetchall():
            out[r["crn"]].append(r["name"])
    return out


def _meeting_tuples(conn, term: str,
                    crns: list[str]) -> dict[str, list[tuple[int, str, str]]]:
    out: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    with cursor(conn) as cur:
        cur.execute(
            "SELECT crn, day_of_week, start_time, end_time FROM meetings "
            "WHERE term_code = %s AND crn = ANY(%s)", (term, crns))
        for r in cur.fetchall():
            out[r["crn"]].append(
                (r["day_of_week"], hhmm(r["start_time"]), hhmm(r["end_time"])))
    return out


# --------------------------------------------------------------------------
# search_sections
# --------------------------------------------------------------------------

def search_sections(conn, query: Optional[str] = None,
                    filters: Optional[SectionFilters] = None,
                    limit: int = 20) -> list[SectionSummary]:
    """Keyword search over sections, narrowed by ``filters``.

    ``query`` may be free text ("machine learning"), a course code ("COMP
    140", "comp140"), or empty to list sections by the filters alone. Free
    text matches on any word (stemmed) in the title, long title or
    description, with title matches weighted highest, plus a literal
    substring match on the titles. An exact course code ranks first. Results
    are ordered by ``score`` (higher is better) and are individual sections;
    collapsed Research placeholders are hidden unless
    ``filters.include_grouped``.

    Semantic (embedding) search is not part of this yet.
    """
    term = resolve_term(conn, filters.term_code if filters else None)
    if term is None:
        return []
    clauses, params = build_filter_clauses(filters, term)

    q = (query or "").strip()
    if not q:
        return query_summaries(
            conn, clauses, params, limit=limit,
            order_by="s.subject_code, s.course_number, s.section_no")

    # Pieces of the relevance expression. Each is added to the score and any
    # one of them matching lets the section into the results.
    extra: dict[str, Any] = {"q_like": f"%{_like(q)}%"}
    match_parts: list[str] = []
    score_parts: list[str] = []

    code = _CODE_RE.match(q)
    if code:
        extra["code_subject"] = code.group(1).upper()
        extra["code_number"] = code.group(2).upper()
        is_code = ("(s.subject_code = %(code_subject)s "
                   "AND upper(s.course_number) = %(code_number)s)")
        match_parts.append(is_code)
        score_parts.append(f"(CASE WHEN {is_code} THEN 10 ELSE 0 END)")

    # OR of one plainto_tsquery per word: any word matches, more words rank
    # higher. Stop words become empty queries and are ignored.
    tokens = re.findall(r"[A-Za-z0-9]+", q)[:8]
    if tokens:
        tsq = " || ".join(
            f"plainto_tsquery('english', %(tok{i})s)" for i in range(len(tokens)))
        extra.update({f"tok{i}": t for i, t in enumerate(tokens)})
        doc = ("(setweight(to_tsvector('english', coalesce(s.title, '')), 'A') "
               "|| setweight(to_tsvector('english', coalesce(s.long_title, '')), 'B') "
               "|| setweight(to_tsvector('english', coalesce(s.description, '')), 'C'))")
        match_parts.append(f"{doc} @@ ({tsq})")
        score_parts.append(f"ts_rank_cd({doc}, ({tsq}))")

    title_like = ("(s.title ILIKE %(q_like)s OR s.long_title ILIKE %(q_like)s)")
    match_parts.append(title_like)
    score_parts.append(f"(CASE WHEN {title_like} THEN 1 ELSE 0 END)")

    clauses.append("(" + " OR ".join(match_parts) + ")")
    return query_summaries(
        conn, clauses, params, limit=limit, extra_params=extra,
        score_sql="(" + " + ".join(score_parts) + ")::float8",
        order_by="score DESC, s.subject_code, s.course_number, s.section_no")


# --------------------------------------------------------------------------
# get_section / get_course_sections
# --------------------------------------------------------------------------

_DETAIL_COLS = [
    "term_code", "crn", "subject_code", "course_number", "section_no", "title",
    "long_title", "description", "department", "part_of_term", "start_date",
    "end_date", "credits", "course_type", "grade_mode", "language",
    "distribution_group", "method", "restrictions", "permission_required",
    "max_enrollment", "enrolled", "waitlisted", "waitlist_max",
    "enrollment_as_of", "final_exam", "meeting_raw", "course_url",
    "prerequisites", "prerequisites_expr", "is_active",
]
_DETAIL_SELECT = ", ".join(f"s.{c}" for c in _DETAIL_COLS)


def _build_details(conn, term: str, rows: list[dict]) -> list[SectionDetail]:
    """Attach meetings and instructors to section rows (all in ``term``)."""
    if not rows:
        return []
    crns = [r["crn"] for r in rows]
    meetings: dict[str, list[Meeting]] = defaultdict(list)
    instructors: dict[str, list[Instructor]] = defaultdict(list)
    with cursor(conn) as cur:
        cur.execute(
            "SELECT crn, day_of_week, start_time, end_time, start_date, "
            "end_date FROM meetings WHERE term_code = %s AND crn = ANY(%s) "
            "ORDER BY day_of_week, start_time, start_date", (term, crns))
        for m in cur.fetchall():
            meetings[m["crn"]].append(Meeting(
                day=m["day_of_week"], day_letter=DAY_LETTERS[m["day_of_week"]],
                start_time=hhmm(m["start_time"]), end_time=hhmm(m["end_time"]),
                start_date=m["start_date"], end_date=m["end_date"]))
        cur.execute(
            "SELECT si.crn, i.key, i.netid, i.name FROM section_instructors si "
            "JOIN instructors i ON i.key = si.instructor_key "
            "WHERE si.term_code = %s AND si.crn = ANY(%s) ORDER BY i.name",
            (term, crns))
        for i in cur.fetchall():
            instructors[i["crn"]].append(
                Instructor(key=i["key"], netid=i["netid"], name=i["name"]))
    return [
        SectionDetail(
            **{c: r[c] for c in _DETAIL_COLS},
            open_seats=_open_seats(r["max_enrollment"], r["enrolled"]),
            meetings=meetings.get(r["crn"], []),
            instructors=instructors.get(r["crn"], []),
        )
        for r in rows
    ]


def get_section(conn, crn: str,
                term_code: Optional[str] = None) -> Optional[SectionDetail]:
    """Full details for one CRN, or ``None`` if it does not exist.

    ``term_code`` defaults to the latest active term. Works for inactive and
    Research sections too (check ``is_active``).
    """
    term = resolve_term(conn, term_code)
    if term is None:
        return None
    with cursor(conn) as cur:
        cur.execute(f"SELECT {_DETAIL_SELECT} FROM sections s "
                    "WHERE s.term_code = %s AND s.crn = %s",
                    (term, str(crn).strip()))
        rows = cur.fetchall()
    details = _build_details(conn, term, rows)
    return details[0] if details else None


def get_course_sections(conn, subject: str, number: str,
                        term_code: Optional[str] = None) -> list[SectionDetail]:
    """Every active section of a course, e.g. ``("COMP", "140")``.

    Returns an empty list if the course is not offered that term. Ordered by
    section number. Research placeholder courses are included here, since
    the caller named the course explicitly.
    """
    term = resolve_term(conn, term_code)
    if term is None:
        return []
    with cursor(conn) as cur:
        cur.execute(
            f"SELECT {_DETAIL_SELECT} FROM sections s "
            "WHERE s.term_code = %s AND s.subject_code = %s "
            "AND upper(s.course_number) = %s AND s.is_active "
            "ORDER BY s.section_no, s.crn",
            (term, subject.strip().upper(), str(number).strip().upper()))
        rows = cur.fetchall()
    return _build_details(conn, term, rows)


# --------------------------------------------------------------------------
# seat_status
# --------------------------------------------------------------------------

def seat_status(conn, crns: Sequence[str],
                term_code: Optional[str] = None) -> list[SeatStatus]:
    """Enrollment, waitlist and data timestamps for each CRN.

    One result per distinct CRN, in the order given. An unknown CRN comes
    back with ``found=False`` instead of being dropped. ``enrollment_as_of``
    is the time Rice reported; ``scraped_at`` is when we fetched it.
    """
    wanted = list(dict.fromkeys(str(c).strip() for c in crns))
    term = resolve_term(conn, term_code)
    if not wanted:
        return []
    found: dict[str, dict] = {}
    if term is not None:
        with cursor(conn) as cur:
            cur.execute(
                "SELECT crn, term_code, enrolled, max_enrollment, waitlisted, "
                "waitlist_max, enrollment_as_of, scraped_at FROM sections "
                "WHERE term_code = %s AND crn = ANY(%s)", (term, wanted))
            found = {r["crn"]: r for r in cur.fetchall()}
    out = []
    for crn in wanted:
        r = found.get(crn)
        if r is None:
            out.append(SeatStatus(crn=crn, found=False))
            continue
        out.append(SeatStatus(
            crn=crn, found=True, term_code=r["term_code"],
            enrolled=r["enrolled"], max_enrollment=r["max_enrollment"],
            open_seats=_open_seats(r["max_enrollment"], r["enrolled"]),
            waitlisted=r["waitlisted"], waitlist_max=r["waitlist_max"],
            enrollment_as_of=r["enrollment_as_of"],
            scraped_at=r["scraped_at"]))
    return out
