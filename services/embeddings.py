"""Course embeddings for semantic search.

One vector per distinct course (subject + number + title) lives in the
``course_embeddings`` table. ``embed_texts`` / ``embed_query`` turn text into
unit-length 384-number vectors with a small local model;
``sync_course_embeddings`` keeps the table up to date; ``search_sections`` in
``services.sections`` compares a query vector with them.

Refresh after each crawl::

    export DATABASE_URL=postgresql://planner:planner@localhost:5433/rice_planner
    python -m services.embeddings

Only new or changed courses are embedded. The first call to the model
downloads it (about 90 MB) and loads it, which takes a few seconds.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .db import cursor

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DIM = 384        # must match course_embeddings.embedding vector(384)

# A function from a list of texts to one unit-length vector per text. Tests
# and other callers can pass their own instead of the default model.
EmbedFn = Callable[[Sequence[str]], list[list[float]]]

_model = None


def get_model():
    """Load the sentence-transformers model once and reuse it."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:   # pragma: no cover - environment problem
            raise ImportError(
                "semantic search needs sentence-transformers; "
                "run `pip install -r requirements.txt`") from e
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Embed ``texts``; every vector has length ``DIM`` and unit length.

    Unit length means cosine similarity is a plain dot product. Text beyond
    the model's 256-token window is ignored.
    """
    if not texts:
        return []
    vectors = get_model().encode(
        list(texts), normalize_embeddings=True, show_progress_bar=False,
        batch_size=32)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed one search query."""
    return embed_texts([text])[0]


def vector_literal(vector: Sequence[float]) -> str:
    """Format a vector as pgvector text, for use as ``%s::vector``.

    Passing text and casting in SQL avoids registering a type adapter on the
    caller's connection.
    """
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _clean(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_course_text(subject_name: Optional[str], subject_code: str,
                      number: str, long_title: Optional[str],
                      description: Optional[str]) -> str:
    """The text that represents a course, e.g.

    ``"Computer Science (COMP 140): COMPUTATIONAL THINKING. <description>"``.

    The subject name helps a query like "computer science courses" find it.
    Facts better handled by filters (instructor, times, credits,
    prerequisites) are deliberately left out.
    """
    head = f"{_clean(subject_name) or subject_code} ({subject_code} {number})"
    title = _clean(long_title)
    desc = _clean(description)
    text = head
    if title:
        text += f": {title}"
        text += "." if not title.endswith((".", "!", "?")) else ""
    elif desc:
        text += ":"
    if desc:
        text += f" {desc}"
    return text


def content_hash(model_name: str, text: str) -> str:
    """Fingerprint of what was embedded (model plus text)."""
    return hashlib.sha256(f"{model_name}\n{text}".encode()).hexdigest()


@dataclass(frozen=True)
class SyncResult:
    embedded: int    # new or changed courses that were embedded
    skipped: int     # courses already up to date
    deleted: int     # stored vectors whose course no longer exists


_COURSES_SQL = """
    SELECT DISTINCT ON (s.subject_code, s.course_number, COALESCE(s.title, ''))
           s.subject_code, s.course_number, COALESCE(s.title, '') AS title,
           s.long_title, s.description, sub.name AS subject_name
    FROM sections s LEFT JOIN subjects sub ON sub.code = s.subject_code
    WHERE s.group_id IS NULL
    ORDER BY s.subject_code, s.course_number, COALESCE(s.title, ''),
             length(COALESCE(s.description, '')) DESC, s.term_code DESC
"""


def sync_course_embeddings(conn, embed_fn: EmbedFn = embed_texts, *,
                           model_name: str = MODEL_NAME,
                           force: bool = False,
                           batch_size: int = 64) -> SyncResult:
    """Make ``course_embeddings`` match the courses in ``sections``.

    Courses are the distinct (subject, number, title) of ordinary sections
    (collapsed Research sections are not embedded). A course is embedded when
    it has no stored vector, its text changed, or it was embedded with a
    different ``model_name``; ``force`` re-embeds all. When a course's
    sections disagree, the longest description is used. Vectors of courses
    that no longer exist are deleted.

    Does not commit; the caller owns the transaction.
    """
    with cursor(conn) as cur:
        cur.execute(_COURSES_SQL)
        courses = cur.fetchall()
        cur.execute("SELECT subject_code, course_number, title, model, "
                    "content_hash FROM course_embeddings")
        stored = {(r["subject_code"], r["course_number"], r["title"]):
                  (r["model"], r["content_hash"]) for r in cur.fetchall()}

        todo, keys = [], set()
        for c in courses:
            key = (c["subject_code"], c["course_number"], c["title"])
            keys.add(key)
            text = build_course_text(c["subject_name"], c["subject_code"],
                                     c["course_number"], c["long_title"],
                                     c["description"])
            digest = content_hash(model_name, text)
            if force or stored.get(key) != (model_name, digest):
                todo.append((key, text, digest))

        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            vectors = embed_fn([t for _, t, _ in batch])
            if len(vectors) != len(batch):
                raise ValueError("embed_fn returned the wrong number of vectors")
            for (key, _, digest), vec in zip(batch, vectors):
                if len(vec) != DIM:
                    raise ValueError(
                        f"embedding has {len(vec)} numbers, expected {DIM}")
                cur.execute(
                    "INSERT INTO course_embeddings (subject_code, "
                    "course_number, title, embedding, model, content_hash, "
                    "updated_at) VALUES (%s, %s, %s, %s::vector, %s, %s, now()) "
                    "ON CONFLICT (subject_code, course_number, title) DO UPDATE "
                    "SET embedding = EXCLUDED.embedding, model = EXCLUDED.model, "
                    "content_hash = EXCLUDED.content_hash, updated_at = now()",
                    (*key, vector_literal(vec), model_name, digest))

        stale = [k for k in stored if k not in keys]
        for key in stale:
            cur.execute("DELETE FROM course_embeddings WHERE subject_code = %s "
                        "AND course_number = %s AND title = %s", key)

    return SyncResult(embedded=len(todo), skipped=len(courses) - len(todo),
                      deleted=len(stale))


def main() -> None:
    """CLI: sync embeddings against ``DATABASE_URL`` and commit."""
    import argparse

    from db.load import connect

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="re-embed every course, even if up to date")
    args = ap.parse_args()
    with connect() as conn:
        result = sync_course_embeddings(conn, force=args.force)
        conn.commit()
    print(f"embedded {result.embedded}, up to date {result.skipped}, "
          f"deleted {result.deleted}")


if __name__ == "__main__":
    main()
