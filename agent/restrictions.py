"""Turn Rice's verbose restriction text into a short, readable summary.

Rice stores a restriction like::

    Must be enrolled in one of the following Level(s):; Graduate;
    Must be enrolled in one of the following Major(s):; English

split by ``"; "``: a header line ending in ``:`` followed by the allowed
values until the next header. ``parse_restrictions`` reads that structure
and ``Restrictions.summary`` gives e.g. ``"Graduate students only; Major:
English"``. Text that does not look like this is kept as-is (the raw text is
always available) rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# "Must be enrolled in one of the following Level(s):" / "Must belong to one
# of the following Cohort(s):" / "Must not be enrolled in ..." (with "(es)").
_HEADER_RE = re.compile(
    r"^Must\s+(?P<neg>not\s+)?(?:be enrolled in|belong to)\s+one of the "
    r"following\s+(?P<kind>.+?)\((?:s|es)\):?$", re.IGNORECASE)

_UNDERGRAD_LEVELS = {"undergraduate", "undergraduate professional",
                     "visiting undergraduate"}

_MAX_SHOWN = 4          # values listed per category before "and N more"


@dataclass(frozen=True)
class Restrictions:
    """Parsed restrictions of one section.

    ``requirements`` maps a category (``"Level"``, ``"Major"``, ``"Program"``,
    ``"Class"``, ``"Degree"``, ``"Cohort"``...) to the allowed values;
    ``excluded`` holds "must not" categories. ``parsed`` is false when the
    text was not in the expected shape, in which case ``summary`` is the raw
    text.
    """

    raw: Optional[str]
    parsed: bool = True
    requirements: dict[str, list[str]] = field(default_factory=dict)
    excluded: dict[str, list[str]] = field(default_factory=dict)

    @property
    def summary(self) -> Optional[str]:
        """One line for people, or ``None`` when there are no restrictions."""
        if not self.raw or not self.raw.strip():
            return None
        if not self.parsed:
            return _clip(" ".join(self.raw.split()), 200)
        parts = []
        for kind, values in self.requirements.items():
            parts.append(_describe(kind, values))
        for kind, values in self.excluded.items():
            parts.append("Not allowed: " + _describe(kind, values))
        return "; ".join(parts) or None


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


def _list(values: list[str]) -> str:
    shown = ", ".join(values[:_MAX_SHOWN])
    extra = len(values) - _MAX_SHOWN
    return f"{shown}, and {extra} more" if extra > 0 else shown


def _describe(kind: str, values: list[str]) -> str:
    if kind.lower() == "level":
        lowered = {v.lower() for v in values}
        if lowered and lowered <= _UNDERGRAD_LEVELS:
            return "Undergraduate students only"
        if lowered == {"graduate"}:
            return "Graduate students only"
    return f"{kind}: {_list(values)}"


def parse_restrictions(raw: Optional[str]) -> Restrictions:
    """Parse Rice restriction text (``None`` or empty means unrestricted)."""
    if raw is None or not raw.strip():
        return Restrictions(raw=raw)
    requirements: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    target: Optional[list[str]] = None
    for piece in (p.strip() for p in raw.split(";")):
        if not piece:
            continue
        header = _HEADER_RE.match(piece)
        if header:
            kind = header.group("kind").strip().title()
            bucket = excluded if header.group("neg") else requirements
            target = bucket.setdefault(kind, [])
        elif target is not None:
            target.append(piece)
        else:                       # a value before any header: unknown shape
            return Restrictions(raw=raw, parsed=False)
    if not (requirements or excluded) or any(
            not v for v in [*requirements.values(), *excluded.values()]):
        return Restrictions(raw=raw, parsed=False)
    return Restrictions(raw=raw, requirements=requirements, excluded=excluded)
