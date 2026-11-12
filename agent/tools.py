"""The tools the model can call, and the code that runs them.

``tool_definitions()`` gives the JSON-schema list sent to Ollama; ``dispatch``
runs one call against the service layer and returns a compact, JSON-safe
"envelope" for the model to read:

    {"term": "202720", "term_label": "Spring 2027", "data_as_of": "...",
     "count": 3, "results": [...], "truncated": false, "notes": [...]}

Design rules enforced here, not left to the model:

* The student's busy blocks live in ``session.busy_blocks`` and are applied
  to every search automatically. No tool accepts busy blocks, so the model
  cannot get them wrong and never computes overlaps; an argument it invents
  is ignored (and reported in ``ignored_arguments``).
* A section with no meeting rows shows ``"NO SCHEDULED MEETING TIME"``
  instead of an empty string.
* Restriction text is shortened to a phrase such as "Graduate students only".
* Every section returned is recorded in ``session.seen`` so the guard can
  tell grounded citations from invented ones.
* A bad call returns ``{"error": ...}`` for the model to read and retry; it
  never raises.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable, Optional

from services import (SectionDetail, SectionFilters, SectionSummary,
                      check_schedule_conflicts, get_course_sections,
                      get_section, list_distribution_groups, list_subjects,
                      search_sections, seat_status, term_label)
from services.db import resolve_term
from services.sections import format_meeting_summary

from . import config
from .restrictions import parse_restrictions
from .session import Session
from .util import format_as_of

NO_MEETING = "NO SCHEDULED MEETING TIME"
MAX_CRNS = 12

# time_of_day -> (start_after, end_before). The code, not the model, decides
# which bound a word means: a small model mixed up "start_after 12:00" with
# "morning".
TIME_OF_DAY = {
    "morning": (None, "12:00"),
    "afternoon": ("12:00", "17:00"),
    "evening": ("17:00", None),
}


class ArgError(ValueError):
    """A tool argument is missing or has the wrong type."""


# --------------------------------------------------------------------------
# Tool schemas
# --------------------------------------------------------------------------

def _fn(name: str, description: str, properties: dict,
        required: Optional[list[str]] = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required or []}}}


_S = {"type": "string"}
_I = {"type": "integer"}
_N = {"type": "number"}
_B = {"type": "boolean"}
_CRNS = {"type": "array", "items": _S,
         "description": f"CRNs as strings, at most {MAX_CRNS}"}

_TOOLS = [
    _fn("search_sections",
        "Search class sections for the current term by topic words, a "
        "course code such as 'COMP 140', or filters alone. Matches meaning "
        "as well as keywords. The student's busy times are already applied "
        "automatically. Use days and time_of_day (or start_after / "
        "end_before for exact times) only for when the student WANTS the "
        "class to meet, e.g. days='MWF' and time_of_day='morning'.",
        {"query": {**_S, "description": "topic words or a course code; "
                   "leave out to list by filters only"},
         "subject": {**_S, "description": "subject code, e.g. COMP, MATH"},
         "distribution_group": {**_S, "description":
                                "e.g. 'Distribution Group I' (or 'I')"},
         "method": {**_S, "description": "e.g. 'Face to Face', 'Online'"},
         "level_min": {**_I, "description": "lowest course number, e.g. 200"},
         "level_max": {**_I, "description": "highest course number, e.g. 299"},
         "instructor": {**_S, "description": "netid or part of a name"},
         "open_seats_only": {**_B, "description":
                             "only sections with seats available"},
         "credits_min": _N, "credits_max": _N,
         "days": {**_S, "description":
                  "days the student WANTS the class to meet, as letters "
                  "M T W R F S U (R = Thursday), e.g. 'MWF' or 'TR'. Only "
                  "sections meeting entirely on these days are returned."},
         "time_of_day": {"type": "string",
                         "enum": list(TIME_OF_DAY), "description":
                         "when the student WANTS the class: 'morning' "
                         "(ends by 12:00), 'afternoon' (12:00-17:00) or "
                         "'evening' (starts 17:00 or later). Prefer this "
                         "over start_after / end_before."},
         "start_after": {**_S, "description":
                         "only for an exact time the student gave: 24-hour "
                         "HH:MM; sections starting at or AFTER it "
                         "('after 1pm' -> '13:00')"},
         "end_before": {**_S, "description":
                        "only for an exact time the student gave: 24-hour "
                        "HH:MM; sections ending at or BEFORE it "
                        "('done by 3pm' -> '15:00')"},
         "limit": {**_I, "description":
                   f"results to return, default {config.MAX_RESULTS}, "
                   f"max {config.MAX_RESULTS_CAP}"}}),
    _fn("get_section",
        "Full details of one section by CRN: description, prerequisites, "
        "meetings, instructors, restrictions, enrollment.",
        {"crn": {**_S, "description": "the section's CRN"}}, ["crn"]),
    _fn("get_course_sections",
        "Every section of one course, e.g. subject 'COMP', number '140'.",
        {"subject": {**_S, "description": "subject code"},
         "number": {**_S, "description": "course number, e.g. '140'"}},
        ["subject", "number"]),
    _fn("check_schedule_conflicts",
        "Check whether a set of candidate sections overlap each other in "
        "time. Always use this instead of comparing times yourself.",
        {"crns": _CRNS}, ["crns"]),
    _fn("seat_status",
        "Current enrollment, seats and waitlist for sections, with the "
        "data timestamp.", {"crns": _CRNS}, ["crns"]),
    _fn("list_subjects",
        "Subject codes and names that have sections this term. Use it to "
        "map a subject name (e.g. 'computer science') to its code.", {}),
    _fn("list_distribution_groups",
        "The distribution group names used by the filter.", {}),
]

_ALLOWED_ARGS = {t["function"]["name"]: set(
    t["function"]["parameters"]["properties"]) for t in _TOOLS}


def tool_definitions() -> list[dict]:
    """The tool list to send to the model."""
    return _TOOLS


# --------------------------------------------------------------------------
# Argument parsing (small models send "300" for 300, "true" for true)
# --------------------------------------------------------------------------

def _str(a: dict, key: str, required: bool = False) -> Optional[str]:
    v = a.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise ArgError(f"missing required argument '{key}'")
        return None
    if isinstance(v, (dict, list)):
        raise ArgError(f"'{key}' must be a string")
    return str(v).strip()


def _int(a: dict, key: str) -> Optional[int]:
    v = a.get(key)
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ArgError(f"'{key}' must be an integer, got {v!r}")
    if f != int(f):
        raise ArgError(f"'{key}' must be an integer, got {v!r}")
    return int(f)


def _num(a: dict, key: str) -> Optional[float]:
    v = a.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ArgError(f"'{key}' must be a number, got {v!r}")


def _bool(a: dict, key: str) -> bool:
    v = a.get(key)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return bool(v)


def _crns(a: dict, key: str = "crns") -> list[str]:
    v = a.get(key)
    if isinstance(v, str):
        v = [p for p in v.replace(",", " ").split() if p]
    if not isinstance(v, list) or not v:
        raise ArgError(f"'{key}' must be a non-empty list of CRNs")
    out = list(dict.fromkeys(str(x).strip() for x in v if str(x).strip()))
    if len(out) > MAX_CRNS:
        raise ArgError(f"at most {MAX_CRNS} CRNs per call, got {len(out)}")
    return out


# --------------------------------------------------------------------------
# Result shaping
# --------------------------------------------------------------------------

def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def record_seen(session: Session, *, term_code: str, crn: str, subject: str,
                number: str, section: str, title: Optional[str],
                permission_required: Optional[bool],
                restrictions: Optional[str], has_meetings: bool,
                enrollment_as_of: Optional[datetime]) -> None:
    """Remember a section a tool returned (what the guard checks against)."""
    session.seen[crn] = {
        "term_code": term_code, "crn": crn, "subject": subject,
        "number": number, "section": section, "title": title,
        "permission_required": bool(permission_required),
        "restrictions": parse_restrictions(restrictions).summary,
        "has_meetings": has_meetings,
        "enrollment_as_of": _iso(enrollment_as_of),
    }


def _meetings_text(has_meetings: bool, text: str) -> str:
    return text if has_meetings and text else NO_MEETING


def compact_summary(session: Session, s: SectionSummary) -> dict:
    """A search result row for the model; also records it as seen."""
    record_seen(session, term_code=s.term_code, crn=s.crn,
                subject=s.subject_code, number=s.course_number,
                section=s.section_no, title=s.title,
                permission_required=s.permission_required,
                restrictions=s.restrictions, has_meetings=s.has_meetings,
                enrollment_as_of=s.enrollment_as_of)
    return {
        "cite_as": f"[CRN {s.crn}]",
        "subject": s.subject_code, "number": s.course_number,
        "section": s.section_no, "crn": s.crn, "title": s.title,
        "credits": s.credits, "instructors": s.instructors or ["not listed"],
        "meetings": _meetings_text(s.has_meetings, s.meeting_summary),
        "open_seats": s.open_seats, "max_enrollment": s.max_enrollment,
        "permission_required": bool(s.permission_required),
        "restrictions": parse_restrictions(s.restrictions).summary,
        "method": s.method, "distribution_group": s.distribution_group,
    }


def compact_detail(session: Session, d: SectionDetail) -> dict:
    """A full section record for the model; also records it as seen."""
    has = bool(d.meetings)
    record_seen(session, term_code=d.term_code, crn=d.crn,
                subject=d.subject_code, number=d.course_number,
                section=d.section_no, title=d.title,
                permission_required=d.permission_required,
                restrictions=d.restrictions, has_meetings=has,
                enrollment_as_of=d.enrollment_as_of)
    text = format_meeting_summary(
        (m.day, m.start_time, m.end_time) for m in d.meetings)
    desc = (d.description or "").strip()
    return {
        "cite_as": f"[CRN {d.crn}]",
        "subject": d.subject_code, "number": d.course_number,
        "section": d.section_no, "crn": d.crn,
        "title": d.long_title or d.title,
        "description": desc if len(desc) <= 600 else desc[:597] + "...",
        "credits": d.credits, "course_type": d.course_type,
        "instructors": [i.name for i in d.instructors] or ["not listed"],
        "meetings": _meetings_text(has, text),
        "meeting_dates": (f"{d.start_date} to {d.end_date}"
                          if d.start_date and d.end_date else None),
        "final_exam": d.final_exam,
        "prerequisites": d.prerequisites,
        "open_seats": d.open_seats, "max_enrollment": d.max_enrollment,
        "waitlisted": d.waitlisted, "waitlist_max": d.waitlist_max,
        "permission_required": bool(d.permission_required),
        "restrictions": parse_restrictions(d.restrictions).summary,
        "restrictions_raw": d.restrictions,
        "method": d.method, "distribution_group": d.distribution_group,
        "language": d.language, "grade_mode": d.grade_mode,
        "is_active": d.is_active,
    }


def _envelope(conn, session: Session, results: list, *,
              as_of: Optional[datetime] = None,
              notes: Optional[list[str]] = None, **extra: Any) -> dict:
    term = resolve_term(conn, session.term_code)
    env = {"term": term, "term_label": term_label(term),
           "data_as_of": format_as_of(as_of), "count": len(results),
           "results": results, "notes": notes or []}
    env.update(extra)
    return env


def _min_as_of(items) -> Optional[datetime]:
    stamps = [x for x in items if x is not None]
    return min(stamps) if stamps else None


def _section_notes(rows: list[dict]) -> list[str]:
    notes = []
    no_time = [r for r in rows if r["meetings"] == NO_MEETING]
    if no_time:
        notes.append(f"{len(no_time)} of these sections have {NO_MEETING}; "
                     "say so when recommending them. Busy times could not "
                     "be checked for them.")
    perm = [r for r in rows if r["permission_required"]]
    if perm:
        notes.append(f"{len(perm)} of these sections require permission to "
                     "enroll; say so when recommending them.")
    if any(r["restrictions"] for r in rows):
        notes.append("Some sections have enrollment restrictions (see "
                     "'restrictions'); mention them when recommending.")
    return notes


# --------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------

def _wanted_times(a: dict) -> tuple[Optional[str], Optional[str]]:
    """``(start_after, end_before)`` from time_of_day and explicit times.

    An explicit time overrides the matching bound of ``time_of_day``.
    """
    start_after, end_before = _str(a, "start_after"), _str(a, "end_before")
    word = _str(a, "time_of_day")
    if word:
        key = word.lower()
        if key not in TIME_OF_DAY:
            raise ArgError(f"'time_of_day' must be one of "
                           f"{', '.join(TIME_OF_DAY)}, got {word!r}")
        default_start, default_end = TIME_OF_DAY[key]
        start_after = start_after or default_start
        end_before = end_before or default_end
    return start_after, end_before


def _describe_meeting_filters(f: SectionFilters) -> str:
    """Readable summary, e.g. ``"days MWF; ends by 12:00"``."""
    parts = []
    if f.days:
        days = f.days if isinstance(f.days, str) else "".join(f.days)
        parts.append(f"days {days.upper()}")
    if f.start_after:
        parts.append(f"starts at or after {f.start_after}")
    if f.end_before:
        parts.append(f"ends by {f.end_before}")
    return "; ".join(parts)


def _search(conn, session: Session, a: dict) -> dict:
    start_after, end_before = _wanted_times(a)
    filters = SectionFilters(
        subject=_str(a, "subject"),
        distribution_group=_str(a, "distribution_group"),
        method=_str(a, "method"),
        level_min=_int(a, "level_min"), level_max=_int(a, "level_max"),
        instructor=_str(a, "instructor"),
        open_seats_only=_bool(a, "open_seats_only"),
        credits_min=_num(a, "credits_min"), credits_max=_num(a, "credits_max"),
        days=_str(a, "days"),
        start_after=start_after, end_before=end_before,
        term_code=session.term_code)
    limit = _int(a, "limit") or config.MAX_RESULTS
    limit = max(1, min(limit, config.MAX_RESULTS_CAP))
    semantic = config.SEMANTIC and not config.SEMANTIC_LOADING
    rows = search_sections(conn, _str(a, "query"), filters, limit=limit + 1,
                           mode="auto" if semantic else "keyword",
                           busy_blocks=session.busy_blocks)
    truncated = len(rows) > limit
    rows = rows[:limit]
    results = [compact_summary(session, s) for s in rows]
    notes = _section_notes(results)
    if session.busy_blocks:
        notes.insert(0, f"Results already exclude sections that overlap the "
                        f"student's {len(session.busy_blocks)} busy time "
                        "block(s).")
    if config.SEMANTIC and config.SEMANTIC_LOADING and _str(a, "query"):
        notes.append("Meaning-based search is still loading, so this used "
                     "keyword matching only; related courses with different "
                     "wording may be missing.")
    if filters.days or filters.start_after or filters.end_before:
        notes.append("Filtered to the requested meeting days/times ("
                     f"{_describe_meeting_filters(filters)}); sections "
                     f"with {NO_MEETING} are left out because they cannot "
                     "be matched.")
    if truncated:
        notes.append(f"More sections match; only the first {limit} are shown.")
    if not results:
        notes.append("No sections matched. Say so; do not suggest sections "
                     "that were not returned.")
    return _envelope(conn, session, results,
                     as_of=_min_as_of(s.enrollment_as_of for s in rows),
                     notes=notes, truncated=truncated,
                     busy_times_applied=bool(session.busy_blocks),
                     meeting_filters=_describe_meeting_filters(filters) or None)


def _get_section(conn, session: Session, a: dict) -> dict:
    crn = _str(a, "crn", required=True)
    d = get_section(conn, crn, session.term_code)
    if d is None:
        return _envelope(conn, session, [],
                         notes=[f"No section with CRN {crn} exists this term."])
    row = compact_detail(session, d)
    return _envelope(conn, session, [row], as_of=d.enrollment_as_of,
                     notes=_section_notes([row]))


def _get_course_sections(conn, session: Session, a: dict) -> dict:
    subject = _str(a, "subject", required=True)
    number = _str(a, "number", required=True)
    details = get_course_sections(conn, subject, number, session.term_code)
    rows = [compact_detail(session, d) for d in details]
    for r in rows:      # keep the list short: the description is repeated
        r.pop("description", None)
        r.pop("restrictions_raw", None)
    notes = _section_notes(rows)
    if not rows:
        notes.append(f"No sections of {subject.upper()} {number} this term.")
    return _envelope(conn, session, rows,
                     as_of=_min_as_of(d.enrollment_as_of for d in details),
                     notes=notes)


def _check_conflicts(conn, session: Session, a: dict) -> dict:
    crns = _crns(a)
    report = check_schedule_conflicts(conn, crns, session.term_code)
    details = {}
    for crn in crns:
        d = get_section(conn, crn, session.term_code)
        if d is not None:
            compact_detail(session, d)      # record as seen
            details[crn] = d
    no_time = [c for c, d in details.items() if not d.meetings]
    notes = []
    if report.conflicts:
        notes.append("These sections overlap in time; do not recommend them "
                     "together.")
    elif not report.unknown_crns:
        notes.append("No time conflicts among the sections that have "
                     "meeting times.")
    if no_time:
        notes.append(f"CRN(s) {', '.join(no_time)} have {NO_MEETING}, so "
                     "they cannot be checked for time conflicts.")
    if report.unknown_crns:
        notes.append(f"Unknown CRN(s): {', '.join(report.unknown_crns)}.")
    return _envelope(
        conn, session,
        [{"crns": [c.crn_a, c.crn_b], "day": c.day_letter,
          "overlap": f"{c.overlap_start}-{c.overlap_end}"}
         for c in report.conflicts],
        as_of=_min_as_of(d.enrollment_as_of for d in details.values()),
        notes=notes, no_conflicts=report.ok,
        unknown_crns=report.unknown_crns, no_meeting_time_crns=no_time)


def _seat_status(conn, session: Session, a: dict) -> dict:
    statuses = seat_status(conn, _crns(a), session.term_code)
    rows = [{"crn": s.crn, "found": s.found, "enrolled": s.enrolled,
             "max_enrollment": s.max_enrollment, "open_seats": s.open_seats,
             "waitlisted": s.waitlisted, "waitlist_max": s.waitlist_max,
             "data_as_of": format_as_of(s.enrollment_as_of)}
            for s in statuses]
    notes = [f"CRN {s.crn} was not found." for s in statuses if not s.found]
    return _envelope(conn, session, rows,
                     as_of=_min_as_of(s.enrollment_as_of for s in statuses),
                     notes=notes)


def _list_subjects(conn, session: Session, a: dict) -> dict:
    subjects = [s for s in list_subjects(conn, session.term_code)
                if s.active_sections > 0]
    return _envelope(
        conn, session,
        [{"code": s.code, "name": s.name, "sections": s.active_sections}
         for s in subjects],
        notes=["Only subjects with sections loaded this term are listed."])


def _list_groups(conn, session: Session, a: dict) -> dict:
    return _envelope(
        conn, session,
        [{"name": g.name, "sections": g.active_sections}
         for g in list_distribution_groups(conn, session.term_code)])


_HANDLERS: dict[str, Callable[[Any, Session, dict], dict]] = {
    "search_sections": _search,
    "get_section": _get_section,
    "get_course_sections": _get_course_sections,
    "check_schedule_conflicts": _check_conflicts,
    "seat_status": _seat_status,
    "list_subjects": _list_subjects,
    "list_distribution_groups": _list_groups,
}


def dispatch(conn, session: Session, name: str, arguments: Any) -> dict:
    """Run tool ``name`` and return its envelope; never raises.

    ``arguments`` may be a dict or a JSON string (some models send strings).
    Arguments the tool does not declare (for example ``busy_blocks``) are
    dropped and listed under ``ignored_arguments``. Every result is also
    scanned so course codes it mentions count as grounded in the session.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"Unknown tool {name!r}. Available tools: "
                         f"{', '.join(_HANDLERS)}."}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {"error": "Tool arguments were not valid JSON."}
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be an object."}
    allowed = _ALLOWED_ARGS[name]
    args = {k: v for k, v in arguments.items() if k in allowed}
    ignored = sorted(set(arguments) - allowed)
    try:
        result = handler(conn, session, args)
    except (ArgError, ValueError) as e:
        return {"error": str(e)}
    except ImportError as e:
        return {"error": f"A required library is missing: {e}"}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:                       # database or driver trouble
        try:
            conn.rollback()
        except Exception:
            pass
        return {"error": f"The lookup failed ({e.__class__.__name__})."}
    if ignored:
        result["ignored_arguments"] = ignored
        result.setdefault("notes", []).append(
            f"Ignored unsupported argument(s): {', '.join(ignored)}. Busy "
            "times are applied automatically.")
    session.note_text(json.dumps(result, default=str))
    return result
