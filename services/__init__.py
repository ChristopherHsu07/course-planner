"""Core service layer: plain, typed query functions over the course database.

Every function takes an open psycopg connection (``db.load.connect()``) as its
first argument, only reads, and returns dataclasses from ``services.models``
that have ``to_dict()`` for JSON. These are the functions an AI agent's tools
will call.
"""

from .models import (BusyBlock, Conflict, ConflictReport, DataFreshness,
                     DistributionGroup, Instructor, Meeting, SeatStatus,
                     SectionDetail, SectionFilters, SectionSummary, Subject)
from .scheduling import check_schedule_conflicts, find_sections_avoiding
from .sections import (get_course_sections, get_section, search_sections,
                       seat_status)
from .vocab import (get_data_freshness, list_distribution_groups,
                    list_subjects, term_label)

__all__ = [
    "search_sections", "find_sections_avoiding", "check_schedule_conflicts",
    "get_section", "get_course_sections", "list_subjects",
    "list_distribution_groups", "seat_status", "get_data_freshness",
    "term_label",
    "BusyBlock", "SectionFilters", "SectionSummary", "SectionDetail",
    "Meeting", "Instructor", "Conflict", "ConflictReport", "SeatStatus",
    "Subject", "DistributionGroup", "DataFreshness",
]
