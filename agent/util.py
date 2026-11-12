"""Small formatting helpers shared by the agent modules."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from . import config


def format_as_of(dt: Optional[datetime]) -> Optional[str]:
    """Human text for a data timestamp, e.g. ``"Nov 7, 2026 3:42 PM CST"``.

    Rice's enrollment timestamps are stored without a time zone and are local
    (Central) time, so a naive value is labelled with ``config.TIMEZONE``.
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(config.TIMEZONE))
    else:
        dt = dt.astimezone(ZoneInfo(config.TIMEZONE))
    hour = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return (f"{dt:%b} {dt.day}, {dt.year} {hour}:{dt.minute:02d} {ampm} "
            f"{dt.tzname()}")
