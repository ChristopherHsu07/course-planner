"""The busy-block overlap condition, shared by every query that accepts one.

``find_sections_avoiding`` and ``search_sections(busy_blocks=...)`` both add
the clause returned by ``busy_clause`` to their WHERE list.
"""

from __future__ import annotations

from typing import Any, Sequence

from .models import BusyBlock


def coerce_blocks(blocks: Sequence[BusyBlock | dict]) -> list[BusyBlock]:
    """``BusyBlock`` objects, or tool-format dicts converted to them.

    Raises ``ValueError`` for an invalid block.
    """
    return [b if isinstance(b, BusyBlock) else BusyBlock.from_dict(b)
            for b in blocks]


def busy_clause(blocks: Sequence[BusyBlock], params: dict[str, Any]) -> str:
    """SQL condition: the section (alias ``s``) meets during none of ``blocks``.

    Adds the ``b_*`` parameters it uses to ``params``. Two meetings clash on
    the same weekday when ``m.start < b.end AND b.start < m.end`` (back-to-back
    is fine) and the date ranges intersect; a missing date is unbounded, so a
    block with no dates covers the whole term. A section with no meetings
    never clashes.
    """
    # Parallel arrays unnested row-by-row into a (day, start, end, ...)
    # table. Casts are explicit so empty/NULL date arrays still type.
    params.update(
        b_days=[b.day_num for b in blocks],
        b_starts=[b.start for b in blocks],
        b_ends=[b.end for b in blocks],
        b_sdates=[b.start_date for b in blocks],
        b_edates=[b.end_date for b in blocks],
    )
    return """NOT EXISTS (
        SELECT 1 FROM meetings m
        JOIN unnest(%(b_days)s::smallint[], %(b_starts)s::time[],
                    %(b_ends)s::time[], %(b_sdates)s::date[],
                    %(b_edates)s::date[])
             AS b(day, start_t, end_t, sdate, edate)
          ON m.day_of_week = b.day
        WHERE m.term_code = s.term_code AND m.crn = s.crn
          AND m.start_time < b.end_t AND b.start_t < m.end_time
          AND (b.sdate IS NULL OR m.end_date   IS NULL OR m.end_date   >= b.sdate)
          AND (b.edate IS NULL OR m.start_date IS NULL OR m.start_date <= b.edate)
    )"""
