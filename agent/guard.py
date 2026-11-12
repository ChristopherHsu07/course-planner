"""Checks and fixes applied to the model's final answer, in code.

The model is asked to follow rules, but a small local model will sometimes
forget. This module makes the important ones hold anyway:

* **Grounding.** Every CRN and course code in the answer must have appeared in
  a tool result (or the student's own message). ``find_ungrounded`` lists the
  ones that did not, so the core can ask the model to fix them once;
  ``finalize`` removes whatever is still unverified and says so.
* **Citations.** The model writes ``[CRN 12345]``; ``finalize`` expands it to
  ``COMP 140 section 002 (CRN 12345)`` from the stored section, so subject,
  number, section and CRN are always present.
* **Flags.** If a cited section needs permission, has restrictions or has no
  scheduled meeting time and the answer does not say so near its mention, a
  line generated from the data is added under "Notes:".
* **Footer.** The data-as-of time and the ESTHER / Registrar reminder are
  added by code to any answer that used tool data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Optional

from services import DataFreshness

from .session import COURSE_CODE_RE, Session
from .util import format_as_of

REMINDER = ("ESTHER and the Registrar are the authoritative sources for "
            "registration; confirm there before you enroll.")

# "[CRN 12345]" (the marker the model is told to use).
_MARKER_RE = re.compile(r"\[\s*CRN[:\s#]*(\d{4,6})\s*\]", re.IGNORECASE)
# "CRN 12345", "CRN: 12345", "CRNs 12345" (a bare mention).
_LABELED_RE = re.compile(r"\bCRNs?[:\s#]*(\d{4,6})\b", re.IGNORECASE)

_PERMISSION_WORDS = re.compile(r"permission|consent|approval", re.IGNORECASE)
_NO_MEETING_WORDS = re.compile(
    r"no scheduled meeting|no meeting|not scheduled|arranged|no set time|"
    r"does not (?:have|meet)|without a (?:set )?(?:meeting )?time|\btba\b",
    re.IGNORECASE)
_RESTRICTION_WORDS = re.compile(
    r"restrict|only|graduate|undergrad|major|program|cohort|class of|"
    r"must be enrolled|limited to", re.IGNORECASE)


def cite(sec: dict[str, Any]) -> str:
    """Full citation for a stored section: ``COMP 140 section 002 (CRN 22130)``."""
    return (f"{sec['subject']} {sec['number']} section {sec['section']} "
            f"(CRN {sec['crn']})")


@dataclass
class GuardResult:
    """The finished answer plus what was checked."""

    text: str                  # the full answer, footer included
    body: str = ""             # the answer without the footer
    cited_crns: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)   # unverified references
    notes_added: int = 0


def find_ungrounded(text: str, session: Session,
                    subjects: Optional[Iterable[str]] = None) -> list[str]:
    """References in ``text`` that no tool result supports.

    Returns strings such as ``"CRN 99999"`` or ``"COMP 999"``. A course code
    is only checked if its subject is in ``subjects`` (when given), so words
    like "FALL 202" are not mistaken for courses.
    """
    bad: list[str] = []
    for m in [*_MARKER_RE.finditer(text), *_LABELED_RE.finditer(text)]:
        crn = m.group(1)
        if crn not in session.seen and f"CRN {crn}" not in bad:
            bad.append(f"CRN {crn}")
    known_subjects = {s.upper() for s in subjects} if subjects else None
    for subject, number in COURSE_CODE_RE.findall(text.upper()):
        if known_subjects is not None and subject not in known_subjects:
            continue
        code = f"{subject} {number}"
        if code not in session.known_courses and code not in bad:
            bad.append(code)
    return bad


def correction_message(bad: list[str]) -> str:
    """What to tell the model when its answer cited unsupported items."""
    return ("Your answer mentioned items that no tool returned: "
            f"{', '.join(bad)}. Rewrite the answer using only sections and "
            "courses from tool results (call a tool if you need more), "
            "citing sections as [CRN n].")


def _expand(text: str, session: Session) -> tuple[str, list[str]]:
    """Replace markers with full citations; drop references with no source."""
    removed: list[str] = []

    def marker(m: re.Match) -> str:
        sec = session.seen.get(m.group(1))
        if sec is None:
            removed.append(f"CRN {m.group(1)}")
            return "[unverified CRN removed]"
        return cite(sec)

    text = _MARKER_RE.sub(marker, text)

    def bare(m: re.Match) -> str:
        crn = m.group(1)
        sec = session.seen.get(crn)
        if sec is None:
            removed.append(f"CRN {crn}")
            return "[unverified CRN removed]"
        # Already cited in full nearby by the model? Leave it alone.
        window = text[max(0, m.start() - 60):m.start()].upper()
        if f"{sec['subject']} {sec['number']}" in window:
            return m.group(0)
        return cite(sec)

    return _LABELED_RE.sub(bare, text), removed


def _segments(text: str) -> dict[str, str]:
    """For each CRN mentioned, its line plus the lines after it that do not
    mention another CRN (a bullet and its continuation lines)."""
    out: dict[str, str] = {}
    current: list[str] = []          # CRNs whose segment is still open
    for line in text.splitlines():
        crns = [m.group(1) for m in _LABELED_RE.finditer(line)]
        if crns:
            current = crns
        for crn in current:
            out[crn] = out.get(crn, "") + " " + line
    return out


def _flag_notes(text: str, session: Session, crns: list[str]) -> list[str]:
    """Lines for facts about cited sections the answer did not mention."""
    segments = _segments(text)
    notes = []
    for crn in crns:
        sec = session.seen[crn]
        seg = segments.get(crn, "")
        problems = []
        if sec["permission_required"] and not _PERMISSION_WORDS.search(seg):
            problems.append("permission is required to enroll")
        if not sec["has_meetings"] and not _NO_MEETING_WORDS.search(seg):
            problems.append("it has no scheduled meeting time")
        if sec["restrictions"] and not _RESTRICTION_WORDS.search(seg):
            problems.append(f"restrictions: {sec['restrictions']}")
        if problems:
            notes.append(f"- {cite(sec)}: " + "; ".join(problems) + ".")
    return notes


def _data_as_of(session: Session, crns: list[str],
                freshness: Optional[DataFreshness]) -> Optional[datetime]:
    """Oldest enrollment timestamp among cited sections, else the session's,
    else the term's."""
    def stamps(keys: Iterable[str]) -> list[datetime]:
        return [datetime.fromisoformat(session.seen[k]["enrollment_as_of"])
                for k in keys if session.seen[k].get("enrollment_as_of")]

    for pool in (crns, list(session.seen)):
        found = stamps(pool)
        if found:
            return min(found)
    return freshness.enrollment_as_of_min if freshness else None


def footer(session: Session, crns: list[str],
           freshness: Optional[DataFreshness]) -> str:
    """The data-freshness line and the registration reminder."""
    as_of = format_as_of(_data_as_of(session, crns, freshness))
    label = freshness.term_label if freshness and freshness.term_label else None
    when = f"Data as of {as_of}" if as_of else "Data timestamp unavailable"
    return f"{when}{f' ({label})' if label else ''}. {REMINDER}"


def finalize(text: str, session: Session,
             freshness: Optional[DataFreshness] = None, *,
             used_tools: bool = False) -> GuardResult:
    """Apply citation expansion, flag completion and the footer to ``text``.

    ``used_tools`` says whether any tool ran this turn; the footer is added
    when it did or when the answer cites a section.
    """
    text = (text or "").strip()
    text, removed = _expand(text, session)
    cited = [c for c in dict.fromkeys(
        m.group(1) for m in _LABELED_RE.finditer(text)) if c in session.seen]

    notes = _flag_notes(text, session, cited)
    if notes:
        text += "\n\nNotes:\n" + "\n".join(notes)
    if removed:
        text += ("\n\nWarning: I removed " + ", ".join(dict.fromkeys(removed))
                 + " because it could not be verified in the database.")
    body = text
    if used_tools or cited:
        text += "\n\n" + footer(session, cited, freshness)
    return GuardResult(text=text, body=body, cited_crns=cited,
                       removed=removed, notes_added=len(notes))


def warn_unverified_courses(text: str, bad_codes: list[str]) -> str:
    """Add a visible warning for course codes that could not be verified."""
    if not bad_codes:
        return text
    return (text + "\n\nWarning: I could not verify "
            + ", ".join(bad_codes) + " in the database.")
