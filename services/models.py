"""Typed inputs and outputs for the service layer.

Everything here is a frozen dataclass from the standard library. Each result
type has ``to_dict()`` that returns plain JSON-safe values (strings, numbers,
lists, dicts), so a later AI tool wrapper can serialize results directly.

Day letters follow Rice: ``M T W R F S U`` (R = Thursday, U = Sunday). In
the database and in ``Meeting.day`` / ``Conflict.day`` they are the ints
1 (Monday) to 7 (Sunday); ``DAY_LETTERS`` and ``DAY_NUM`` convert.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from typing import Any, Optional

DAY_NUM = {"M": 1, "T": 2, "W": 3, "R": 4, "F": 5, "S": 6, "U": 7}
DAY_LETTERS = {v: k for k, v in DAY_NUM.items()}

_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def parse_hhmm(value: str | time) -> time:
    """Turn ``"10:00"`` (24-hour ``H:MM`` or ``HH:MM``) into a ``time``.

    Raises ``ValueError`` for anything else, including ``"10:00 AM"``.
    """
    if isinstance(value, time):
        return value
    m = _TIME_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"time must look like 'HH:MM' (24-hour), got {value!r}")
    return time(int(m.group(1)), int(m.group(2)))


def _parse_date(value: str | date | None) -> Optional[date]:
    if value is None or isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"date must look like 'YYYY-MM-DD', got {value!r}")


def hhmm(t: Optional[time]) -> Optional[str]:
    """Format a ``time`` as ``"HH:MM"`` (``None`` stays ``None``)."""
    return t.strftime("%H:%M") if t is not None else None


def _jsonable(value: Any) -> Any:
    """Make dataclass ``asdict`` output JSON-safe (dates, times, nested)."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, time):
        return hhmm(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class _Model:
    def to_dict(self) -> dict[str, Any]:
        """Plain, JSON-safe dict of this object."""
        return _jsonable(asdict(self))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BusyBlock(_Model):
    """A recurring weekly time the student is unavailable.

    ``day`` is a Rice letter (``"M"``); ``start``/``end`` are ``"HH:MM"``
    24-hour strings (``datetime.time`` also accepted). The optional
    ``start_date``/``end_date`` (``"YYYY-MM-DD"`` or ``date``) limit the block
    to part of the term; leave both ``None`` for the whole term. Construction
    validates and raises ``ValueError``; the stored values are normalized
    to ``time`` / ``date``.
    """

    day: str
    start: time
    end: time
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    def __post_init__(self):
        day = str(self.day).strip().upper()
        if day not in DAY_NUM:
            raise ValueError(
                f"day must be one of {' '.join(DAY_NUM)} (R = Thursday), "
                f"got {self.day!r}")
        start, end = parse_hhmm(self.start), parse_hhmm(self.end)
        if not start < end:
            raise ValueError(f"start must be before end, got {start}-{end}")
        sd, ed = _parse_date(self.start_date), _parse_date(self.end_date)
        if sd and ed and sd > ed:
            raise ValueError("start_date must not be after end_date")
        # Frozen dataclass: assign normalized values via object.__setattr__.
        object.__setattr__(self, "day", day)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "start_date", sd)
        object.__setattr__(self, "end_date", ed)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BusyBlock":
        """Build from the tool format ``{"day": "M", "start": "10:00",
        "end": "11:15"}`` (optional ``start_date``/``end_date``)."""
        return cls(d["day"], d["start"], d["end"],
                   d.get("start_date"), d.get("end_date"))

    @property
    def day_num(self) -> int:
        return DAY_NUM[self.day]


@dataclass(frozen=True)
class SectionFilters(_Model):
    """Optional narrowing for ``search_sections`` and ``find_sections_avoiding``.

    * ``subject``: subject code(s), e.g. ``"COMP"`` or ``["COMP", "MATH"]``.
    * ``distribution_group``: ``"Distribution Group I"``; the short forms
      ``"I"``, ``"II"``, ``"III"`` are accepted. Case-insensitive.
    * ``method``: e.g. ``"Face to Face"`` or ``"Online"``. Case-insensitive.
    * ``level_min`` / ``level_max``: inclusive course-number range, e.g.
      300 and 399 for 300-level. Compared to the leading digits of the number.
    * ``instructor``: a netid, or part of a name.
    * ``open_seats_only``: only sections with ``max_enrollment > enrolled``.
    * ``credits_min`` / ``credits_max``: a section matches if any credit
      value it allows lies in the range (``"1 TO 4"`` allows 1 through 4).
    * ``term_code``: defaults to the latest active term.
    * ``include_grouped``: also return collapsed Research placeholder
      sections (off by default).
    """

    subject: Optional[str | list[str]] = None
    distribution_group: Optional[str] = None
    method: Optional[str] = None
    level_min: Optional[int] = None
    level_max: Optional[int] = None
    instructor: Optional[str] = None
    open_seats_only: bool = False
    credits_min: Optional[float] = None
    credits_max: Optional[float] = None
    term_code: Optional[str] = None
    include_grouped: bool = False


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Meeting(_Model):
    """One weekly meeting. ``day`` is 1 (Mon) to 7 (Sun); ``day_letter`` M-U."""

    day: int
    day_letter: str
    start_time: str          # "HH:MM"
    end_time: str            # "HH:MM"
    start_date: Optional[date]
    end_date: Optional[date]


@dataclass(frozen=True)
class SectionSummary(_Model):
    """A section as shown in search results."""

    term_code: str
    crn: str
    subject_code: str
    course_number: str
    section_no: str
    title: Optional[str]
    credits: Optional[str]
    distribution_group: Optional[str]
    method: Optional[str]
    course_type: Optional[str]
    instructors: list[str]
    meeting_summary: str     # e.g. "MW 14:00-15:15"; "" if no meetings
    max_enrollment: Optional[int]
    enrolled: Optional[int]
    open_seats: Optional[int]
    score: Optional[float] = None   # search rank; None outside search


@dataclass(frozen=True)
class Instructor(_Model):
    key: str
    netid: Optional[str]
    name: str


@dataclass(frozen=True)
class SectionDetail(_Model):
    """Everything known about one section."""

    term_code: str
    crn: str
    subject_code: str
    course_number: str
    section_no: str
    title: Optional[str]
    long_title: Optional[str]
    description: Optional[str]
    department: Optional[str]
    part_of_term: Optional[str]
    start_date: Optional[date]
    end_date: Optional[date]
    credits: Optional[str]
    course_type: Optional[str]
    grade_mode: Optional[str]
    language: Optional[str]
    distribution_group: Optional[str]
    method: Optional[str]
    restrictions: Optional[str]
    permission_required: Optional[bool]
    max_enrollment: Optional[int]
    enrolled: Optional[int]
    open_seats: Optional[int]
    waitlisted: Optional[int]
    waitlist_max: Optional[int]
    enrollment_as_of: Optional[datetime]
    final_exam: Optional[str]
    meeting_raw: Optional[str]
    course_url: Optional[str]
    prerequisites: Optional[str]
    prerequisites_expr: Optional[Any]
    is_active: Optional[bool]
    meetings: list[Meeting] = field(default_factory=list)
    instructors: list[Instructor] = field(default_factory=list)


@dataclass(frozen=True)
class Conflict(_Model):
    """Two sections meeting at the same time on the same day."""

    crn_a: str
    crn_b: str
    day: int
    day_letter: str
    overlap_start: str       # "HH:MM"
    overlap_end: str         # "HH:MM"


@dataclass(frozen=True)
class ConflictReport(_Model):
    """Result of ``check_schedule_conflicts``.

    ``ok`` is true only when every CRN was found and there are no conflicts.
    """

    ok: bool
    conflicts: list[Conflict]
    unknown_crns: list[str]


@dataclass(frozen=True)
class SeatStatus(_Model):
    """Current enrollment for one CRN. ``found`` is false for unknown CRNs."""

    crn: str
    found: bool
    term_code: Optional[str] = None
    enrolled: Optional[int] = None
    max_enrollment: Optional[int] = None
    open_seats: Optional[int] = None
    waitlisted: Optional[int] = None
    waitlist_max: Optional[int] = None
    enrollment_as_of: Optional[datetime] = None   # Rice's own timestamp
    scraped_at: Optional[datetime] = None         # when we fetched it


@dataclass(frozen=True)
class Subject(_Model):
    code: str
    name: str
    active_sections: int


@dataclass(frozen=True)
class DistributionGroup(_Model):
    name: str
    active_sections: int
