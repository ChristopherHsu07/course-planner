"""Vocabulary lists the agent uses to map user language to database codes.

``list_subjects`` ("computer science" -> ``COMP``) and
``list_distribution_groups`` ("humanities" -> one of the group names) return
the exact strings the filters accept.
"""

from __future__ import annotations

from typing import Optional

from .db import cursor, resolve_term
from .models import DistributionGroup, Subject


def list_subjects(conn, term_code: Optional[str] = None) -> list[Subject]:
    """Every subject code and name, with its active-section count for the term.

    Subjects that are known but have no sections loaded (not crawled yet, or
    not offered) are included with a count of 0. Ordered by code.
    """
    term = resolve_term(conn, term_code)
    with cursor(conn) as cur:
        cur.execute(
            "SELECT sub.code, sub.name, count(s.crn)::int AS n "
            "FROM subjects sub LEFT JOIN sections s "
            "ON s.subject_code = sub.code AND s.is_active "
            "AND s.term_code = %s GROUP BY sub.code, sub.name "
            "ORDER BY sub.code", (term,))
        return [Subject(code=r["code"], name=r["name"], active_sections=r["n"])
                for r in cur.fetchall()]


def list_distribution_groups(conn, term_code: Optional[str] = None
                             ) -> list[DistributionGroup]:
    """Distinct distribution group names with their active-section counts.

    Names are exactly as stored (e.g. ``"Distribution Group I"``). Sections
    with no group are not listed. Ordered by name.
    """
    term = resolve_term(conn, term_code)
    with cursor(conn) as cur:
        cur.execute(
            "SELECT distribution_group AS name, count(*)::int AS n "
            "FROM sections WHERE is_active AND term_code = %s "
            "AND distribution_group IS NOT NULL AND distribution_group <> '' "
            "GROUP BY 1 ORDER BY 1", (term,))
        return [DistributionGroup(name=r["name"], active_sections=r["n"])
                for r in cur.fetchall()]
