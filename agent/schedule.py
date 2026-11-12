"""Turn what the student says about their schedule into structured busy blocks.

This runs as its own step *before* the main model and its tools. The model
extracts blocks with a JSON-schema constrained reply (Ollama structured
output); this module then validates every block through ``BusyBlock`` and
stores them in the session. The main model never sees raw times to reason
about, never passes blocks to a tool, and never computes an overlap: the
database does that.

If a time is genuinely ambiguous the extractor asks a question instead of
guessing, and the turn stops there (no tools run on a guess).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from services import BusyBlock

from .llm import LLM

DAY_NAMES = "MTWRFSU"

# Reply shape the extractor must follow.
SCHEMA = {
    "type": "object",
    "properties": {
        "busy_blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "string", "enum": list(DAY_NAMES)},
                    "start": {"type": "string",
                              "description": "24-hour HH:MM"},
                    "end": {"type": "string", "description": "24-hour HH:MM"},
                },
                "required": ["day", "start", "end"],
            },
        },
        "clarification": {
            "type": "string",
            "description": "a question for the student, or an empty string",
        },
    },
    "required": ["busy_blocks", "clarification"],
}

SYSTEM_PROMPT = """\
You extract a student's weekly busy times from their message into JSON.

Rules:
- Return the COMPLETE updated list of busy blocks: keep every current block
  unless the student removes or changes it. If they say they have nothing
  scheduled or want to start over, return an empty list.
- Days are single letters: M=Monday T=Tuesday W=Wednesday R=Thursday
  F=Friday S=Saturday U=Sunday. "MWF" is three blocks (M, W, F). "TR",
  "TTh" and "Tue/Thu" mean T and R. Make one block per day.
- Times are 24-hour "HH:MM". Convert AM/PM: 2:30 PM -> "14:30", 12 PM ->
  "12:00", 12 AM -> "00:00". "All day" means "00:00" to "23:59".
- Only ask a question (put it in "clarification" and leave the blocks
  unchanged) when you truly cannot tell: a time with no AM/PM that could be
  either (for example "10 to 11"), a vague period like "mornings" or "after
  lunch" with no times, or a day that is missing. Never guess. Otherwise
  leave "clarification" as an empty string.
- Do not invent blocks. If the message says nothing about busy times, return
  the current blocks unchanged.
- Blocks are ONLY times the student is unavailable (class, work, practice).
  When the student asks for a class that meets at certain days or times
  ("find a class on MWF mornings"), those are times they WANT, not busy
  times: return the current blocks unchanged and an empty clarification.

Example 1
Current blocks: []
Message: "I have class Monday and Wednesday 10:00am to 11:15am"
Output: {"busy_blocks":[{"day":"M","start":"10:00","end":"11:15"},
{"day":"W","start":"10:00","end":"11:15"}],"clarification":""}

Example 2
Current blocks: [{"day":"M","start":"10:00","end":"11:15"}]
Message: "I also work Fridays from 1 to 5"
Output: {"busy_blocks":[{"day":"M","start":"10:00","end":"11:15"}],
"clarification":"Is your Friday work 1 to 5 in the afternoon (1:00 PM to 5:00 PM)?"}

Example 3
Current blocks: []
Message: "Give me a class that meets MWF mornings"
Output: {"busy_blocks":[],"clarification":""}
"""

# Words and patterns that suggest a message talks about the student's time.
# A message with none of them skips the extraction call entirely.
_SCHEDULE_HINT = re.compile(
    r"\d\s*[:.]\s*\d\d|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)|"
    r"\b(?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[a-z]*\b|"
    r"\b(?:mwf|mw|tr|tth|mtwrf)\b|\b(?:morning|afternoon|evening|noon|lunch|"
    r"night)s?\b|\b(?:busy|free|available|unavailable|my schedule|"
    r"commitments?|start over|clear my|reset|no longer|forget my|"
    r"remove my|instead)\b|\b(?:before|after|between|from|until|till|at)\s+"
    r"\d", re.IGNORECASE)


# "Find me a class that meets MWF mornings" is about the class the student
# WANTS, not times they are busy. It mentions days and "mornings", so it
# would trip the hint above; this pair of patterns lets it through to the
# search tool (which has days / time_of_day / start_after / end_before
# filters) instead.
_REQUEST = re.compile(
    r"\b(?:find|give me|show me|looking for|look for|recommend|suggest|"
    r"search for|list|are there|is there|do you have|i (?:want|need|would "
    r"like)|i'd like|what|which)\b", re.IGNORECASE)
_CLASS_WORD = re.compile(r"\b(?:class|classes|course|courses|section|sections|"
                         r"seminar|seminars|lecture|lectures)\b", re.IGNORECASE)
_BUSY_STATEMENT = re.compile(
    r"\bi (?:have|got|work|take|teach)\b|\bi'?ve got\b|"
    r"\b(?:busy|unavailable|can'?t|cannot|not free|my schedule|my classes|"
    r"working|commitments?|also|start over|clear|reset|no longer|forget|"
    r"remove|instead)\b", re.IGNORECASE)


# An exact clock time: "10:00", "2pm", "from 9", "10 to 11".
_CLOCK_TIME = re.compile(
    r"\d\s*[:.]\s*\d\d|\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)|"
    r"\b(?:before|after|between|from|until|till|at)\s+\d|"
    r"\b\d{1,2}\s*(?:-|to)\s*\d{1,2}\b", re.IGNORECASE)


def is_class_request(text: str) -> bool:
    """True when ``text`` asks for classes and says nothing about being busy.

    It needs a class word ("class", "course", ...) and no busy-time
    statement ("I have", "I work", "busy", ...), plus either a request verb
    ("find", "give me", ...) or no exact clock time. The second case covers
    "asian studies class in the morning on MWF", which has no verb; a terse
    "class Monday 10 to 11am" has exact times, so it is still read as busy.
    """
    return bool(_CLASS_WORD.search(text)
                and not _BUSY_STATEMENT.search(text)
                and (_REQUEST.search(text) or not _CLOCK_TIME.search(text)))


def looks_like_schedule(text: str) -> bool:
    """Cheap check: does ``text`` plausibly describe times the student is busy?"""
    return bool(_SCHEDULE_HINT.search(text)) and not is_class_request(text)


@dataclass(frozen=True)
class ScheduleUpdate:
    """Outcome of one extraction step.

    ``blocks`` is the full list to store (the old list when nothing changed
    or when a question is pending). ``clarification`` is a question to show
    the student instead of continuing the turn.
    """

    blocks: list[BusyBlock]
    clarification: Optional[str] = None
    ran: bool = False          # whether the model was called


def describe_blocks(blocks: Sequence[BusyBlock]) -> str:
    """Readable list, e.g. ``"M 10:00-11:15, W 10:00-11:15"``."""
    if not blocks:
        return "none"
    return ", ".join(f"{b.day} {b.start:%H:%M}-{b.end:%H:%M}" for b in blocks)


def _sorted_unique(blocks: Sequence[BusyBlock]) -> list[BusyBlock]:
    unique = {(b.day_num, b.start, b.end, b.start_date, b.end_date): b
              for b in blocks}
    return [unique[k] for k in sorted(
        unique, key=lambda k: (k[0], k[1], k[2]))]


def extract_busy_blocks(llm: LLM, current: Sequence[BusyBlock],
                        message: str) -> ScheduleUpdate:
    """Update ``current`` busy blocks from ``message``.

    Skips the model call (and returns ``current``) when the message does not
    look like it mentions times. Otherwise asks the model for the complete
    updated list as structured JSON and validates every block; an invalid or
    unreadable reply becomes a clarification question rather than a guess.
    """
    current = list(current)
    if not looks_like_schedule(message):
        return ScheduleUpdate(blocks=current)

    shown = [{"day": b.day, "start": f"{b.start:%H:%M}",
              "end": f"{b.end:%H:%M}"} for b in current]
    reply = llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content":
          f"Current blocks: {json.dumps(shown)}\nMessage: {message}"}],
        format=SCHEMA, think=False)

    retry = ("I could not read your schedule from that. Could you list the "
             "days and exact start and end times, for example "
             "\"Mon/Wed 10:00 AM to 11:15 AM\"?")
    try:
        data = json.loads(reply.get("content") or "")
        raw_blocks = data["busy_blocks"]
        question = (data.get("clarification") or "").strip()
        if not isinstance(raw_blocks, list):
            raise ValueError("busy_blocks is not a list")
    except (ValueError, KeyError, TypeError):
        return ScheduleUpdate(blocks=current, clarification=retry, ran=True)

    if question:
        return ScheduleUpdate(blocks=current, clarification=question, ran=True)

    blocks: list[BusyBlock] = []
    for raw in raw_blocks:
        try:
            blocks.append(BusyBlock.from_dict(raw))
        except (ValueError, KeyError, TypeError, AttributeError):
            return ScheduleUpdate(blocks=current, clarification=retry, ran=True)
    return ScheduleUpdate(blocks=_sorted_unique(blocks), ran=True)
