"""Schedule-aware queries: ``find_sections_avoiding`` and
``check_schedule_conflicts``.

Two meetings clash when they are on the same weekday, their times overlap
(``a.start < b.end AND b.start < a.end``, so back-to-back is fine), and
their date ranges intersect (so a first-half-term class does not clash with a
second-half-term one). A missing date counts as unbounded.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from .db import cursor, resolve_term
from .models import (DAY_LETTERS, BusyBlock, Conflict, ConflictReport,
                     SectionFilters, SectionSummary, hhmm)
from .busy import coerce_blocks
from .sections import search_sections


def find_sections_avoiding(conn, busy_blocks: Sequence[BusyBlock | dict],
                           filters: Optional[SectionFilters] = None,
                           limit: int = 50) -> list[SectionSummary]:
    """Sections with no meeting overlapping any busy block.

    ``busy_blocks`` are ``BusyBlock`` objects or dicts such as ``{"day":
    "M", "start": "10:00", "end": "11:15"}`` (optionally with ``start_date``
    / ``end_date``; without them a block applies to the whole term). A bad
    block raises ``ValueError``. ``filters`` narrow the result exactly as in
    ``search_sections``. A section with no meetings (research, arranged
    time) never conflicts and is included. An empty block list returns the
    filtered sections unchanged.
    """
    # The query itself is search_sections with no text and the blocks applied.
    return search_sections(conn, None, filters, limit,
                           busy_blocks=coerce_blocks(busy_blocks))


def check_schedule_conflicts(conn, crns: Sequence[str],
                             term_code: Optional[str] = None
                             ) -> ConflictReport:
    """Check a candidate set of sections against each other.

    Returns every pair of the given CRNs that meet at overlapping times on
    the same day with intersecting date ranges (one ``Conflict`` per pair
    per day per overlapping meeting pair). CRNs not found in the term are
    listed in ``unknown_crns`` and excluded from the check. Duplicate CRNs
    are counted once. ``ok`` is true only when there are no conflicts and no
    unknown CRNs.
    """
    wanted = list(dict.fromkeys(str(c).strip() for c in crns))
    term = resolve_term(conn, term_code)
    if not wanted:
        return ConflictReport(ok=True, conflicts=[], unknown_crns=[])
    if term is None:
        return ConflictReport(ok=False, conflicts=[], unknown_crns=wanted)

    with cursor(conn) as cur:
        cur.execute("SELECT crn FROM sections WHERE term_code = %s "
                    "AND crn = ANY(%s)", (term, wanted))
        known = {r["crn"] for r in cur.fetchall()}
        unknown = [c for c in wanted if c not in known]

        cur.execute(
            """
            SELECT a.crn AS crn_a, b.crn AS crn_b, a.day_of_week AS day,
                   greatest(a.start_time, b.start_time) AS overlap_start,
                   least(a.end_time, b.end_time) AS overlap_end
            FROM meetings a
            JOIN meetings b
              ON b.term_code = a.term_code AND b.day_of_week = a.day_of_week
             AND a.crn < b.crn
            WHERE a.term_code = %(term)s
              AND a.crn = ANY(%(crns)s) AND b.crn = ANY(%(crns)s)
              AND a.start_time < b.end_time AND b.start_time < a.end_time
              AND (a.end_date IS NULL OR b.start_date IS NULL
                   OR a.end_date >= b.start_date)
              AND (b.end_date IS NULL OR a.start_date IS NULL
                   OR b.end_date >= a.start_date)
            GROUP BY a.crn, b.crn, a.day_of_week,
                     greatest(a.start_time, b.start_time),
                     least(a.end_time, b.end_time)
            ORDER BY a.crn, b.crn, a.day_of_week, overlap_start
            """,
            {"term": term, "crns": sorted(known)})
        conflicts = [
            Conflict(crn_a=r["crn_a"], crn_b=r["crn_b"], day=r["day"],
                     day_letter=DAY_LETTERS[r["day"]],
                     overlap_start=hhmm(r["overlap_start"]),
                     overlap_end=hhmm(r["overlap_end"]))
            for r in cur.fetchall()
        ]
    return ConflictReport(ok=not conflicts and not unknown,
                          conflicts=conflicts, unknown_crns=unknown)
