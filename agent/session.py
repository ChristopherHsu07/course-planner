"""Conversation state for one student.

``Session`` holds everything that must survive between turns: the chat
history, the student's busy blocks (parsed once, outside the main model's
control), the sections the tools have returned (the only things the model may
recommend), and which course codes have appeared in tool results. It is
plain data and round-trips through JSON, so a server could keep it between
requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from services import BusyBlock

# "COMP 140", "MATH 101": four capitals, a space (optional), three digits.
COURSE_CODE_RE = re.compile(r"\b([A-Z]{3,5})\s?(\d{3})[A-Z]?\b")


@dataclass
class Session:
    """State for one conversation (see module docstring)."""

    messages: list[dict] = field(default_factory=list)
    busy_blocks: list[BusyBlock] = field(default_factory=list)
    term_code: Optional[str] = None
    # crn -> facts about a section a tool returned (see tools.record_seen).
    seen: dict[str, dict[str, Any]] = field(default_factory=dict)
    # "COMP 140" strings that appeared in tool results or user messages.
    known_courses: set[str] = field(default_factory=set)
    # Set by the core at the start of each turn.
    tools_called_this_turn: int = 0

    def note_text(self, text: str) -> None:
        """Remember every course code mentioned in ``text``."""
        for subject, number in COURSE_CODE_RE.findall(text.upper()):
            self.known_courses.add(f"{subject} {number}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe form of the whole session."""
        return {
            "messages": self.messages,
            "busy_blocks": [b.to_dict() for b in self.busy_blocks],
            "term_code": self.term_code,
            "seen": self.seen,
            "known_courses": sorted(self.known_courses),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Session":
        return cls(
            messages=list(d.get("messages", [])),
            busy_blocks=[BusyBlock.from_dict(b)
                         for b in d.get("busy_blocks", [])],
            term_code=d.get("term_code"),
            seen=dict(d.get("seen", {})),
            known_courses=set(d.get("known_courses", [])),
        )
