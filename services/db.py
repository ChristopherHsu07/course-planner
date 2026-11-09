"""Database helpers shared by the service functions.

Every service function takes an open psycopg connection as its first
argument and never commits or closes it, the same convention as
``db/load.py``. Open one with ``db.load.connect()``.
"""

from __future__ import annotations

from typing import Optional

from psycopg.rows import dict_row


def cursor(conn):
    """A cursor whose rows are dicts keyed by column name."""
    return conn.cursor(row_factory=dict_row)


def resolve_term(conn, term_code: Optional[str]) -> Optional[str]:
    """Return ``term_code``, or the latest term with an active section.

    Returns ``None`` when the database has no active sections at all.
    """
    if term_code:
        return term_code
    with cursor(conn) as cur:
        cur.execute("SELECT max(term_code) AS term FROM sections "
                    "WHERE is_active")
        return cur.fetchone()["term"]
