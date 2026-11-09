"""Tests for course embeddings and semantic / hybrid search.

Needs ``DATABASE_URL`` and, for the quality tests, vectors in
``course_embeddings`` (``python -m services.embeddings``). The first test
that uses the real model loads it, which takes a few seconds.

Tests that write (the sync tests) use their own connection and roll back, so
the real table is never changed. Everything else is read-only.
"""

import hashlib
import math
import os
import random

import psycopg
import pytest
from psycopg.rows import dict_row

from services import SectionFilters, search_sections
from services.embeddings import (DIM, MODEL_NAME, build_course_text,
                                 embed_query, embed_texts,
                                 sync_course_embeddings, vector_literal)
from services.sections import MIN_SIMILARITY

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL is not set; point it at the Docker database")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as c:
        yield c


@pytest.fixture
def rollback_conn():
    """A connection whose changes are discarded at the end of the test."""
    c = psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row)
    try:
        yield c
    finally:
        c.rollback()
        c.close()


@pytest.fixture(scope="module")
def have_vectors(conn):
    n = q(conn, "SELECT count(*) AS n FROM course_embeddings WHERE model = %s",
          (MODEL_NAME,))[0]["n"]
    if not n:
        pytest.skip("run `python -m services.embeddings` first")
    return n


def fake_embed(texts):
    """Deterministic unit vectors derived from the text (no model needed)."""
    out = []
    for t in texts:
        rng = random.Random(hashlib.sha256(t.encode()).digest())
        v = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = math.sqrt(sum(x * x for x in v))
        out.append([x / norm for x in v])
    return out


def courses(rows):
    """Distinct (course, title) in order of first appearance."""
    seen = []
    for r in rows:
        k = (f"{r.subject_code} {r.course_number}", r.title)
        if k not in seen:
            seen.append(k)
    return seen


def distinct_courses(conn):
    return q(conn, """SELECT count(*) AS n FROM (
        SELECT DISTINCT subject_code, course_number, COALESCE(title, '')
        FROM sections WHERE group_id IS NULL) x""")[0]["n"]


# --------------------------------------------------------------------------
# Text and vectors
# --------------------------------------------------------------------------

def test_build_course_text_has_subject_code_title_and_description():
    t = build_course_text("Computer Science", "COMP", "140",
                          "COMPUTATIONAL THINKING", "An intro\n  to coding.")
    assert t == ("Computer Science (COMP 140): COMPUTATIONAL THINKING. "
                 "An intro to coding.")


def test_build_course_text_handles_missing_parts():
    assert build_course_text(None, "COMP", "140", None, None) == "COMP (COMP 140)"
    assert build_course_text(None, "COMP", "140", "T.", "d") == "COMP (COMP 140): T. d"
    assert build_course_text("X", "COMP", "1", None, "d") == "X (COMP 1): d"


def test_vector_literal_format():
    assert vector_literal([1, 0.5, -2]) == "[1.0,0.5,-2.0]"


def test_real_model_vectors_are_unit_length_and_meaningful():
    a, b, c = embed_texts(["machine learning from data",
                           "teaching computers to learn patterns",
                           "baking sourdough bread"])
    assert len(a) == DIM
    for v in (a, b, c):
        assert math.isclose(sum(x * x for x in v), 1.0, abs_tol=1e-4)
    dot = lambda u, w: sum(x * y for x, y in zip(u, w))
    assert dot(a, b) > dot(a, c)          # related text is closer
    assert embed_texts([]) == []
    assert embed_query("hello") == pytest.approx(embed_texts(["hello"])[0])


# --------------------------------------------------------------------------
# sync_course_embeddings (fake embedder, rolled back)
# --------------------------------------------------------------------------

def test_sync_embeds_everything_then_nothing(rollback_conn):
    n = distinct_courses(rollback_conn)
    first = sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    assert (first.embedded, first.skipped) == (n, 0)
    assert q(rollback_conn, "SELECT count(*) AS n FROM course_embeddings "
             "WHERE model = 'fake'")[0]["n"] == n
    # The real model's rows were replaced by the fake model in this transaction.
    second = sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    assert (second.embedded, second.skipped, second.deleted) == (0, n, 0)


def test_sync_reembeds_only_the_changed_course(rollback_conn):
    sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    row = q(rollback_conn, """SELECT subject_code, course_number
        FROM sections WHERE group_id IS NULL ORDER BY crn LIMIT 1""")[0]
    rollback_conn.execute(
        "UPDATE sections SET description = COALESCE(description, '') || "
        "' Now with a new unit on kites.' WHERE subject_code = %s "
        "AND course_number = %s", (row["subject_code"], row["course_number"]))
    changed = q(rollback_conn, """SELECT count(DISTINCT COALESCE(title, '')) AS n
        FROM sections WHERE subject_code = %s AND course_number = %s""",
                (row["subject_code"], row["course_number"]))[0]["n"]
    r = sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    assert r.embedded == changed and r.deleted == 0


def test_sync_reembeds_all_on_model_change_or_force(rollback_conn):
    n = distinct_courses(rollback_conn)
    sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    other = sync_course_embeddings(rollback_conn, fake_embed, model_name="fake2")
    assert other.embedded == n
    forced = sync_course_embeddings(rollback_conn, fake_embed,
                                    model_name="fake2", force=True)
    assert forced.embedded == n


def test_sync_deletes_vectors_of_missing_courses(rollback_conn):
    sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    rollback_conn.execute(
        "INSERT INTO course_embeddings (subject_code, course_number, title, "
        "embedding, model, content_hash) VALUES ('ZZZZ', '999', 'GONE', %s, "
        "'fake', 'x')", (vector_literal(fake_embed(["x"])[0]),))
    r = sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    assert (r.embedded, r.deleted) == (0, 1)


def test_sync_rejects_wrong_dimension(rollback_conn):
    with pytest.raises(ValueError):
        sync_course_embeddings(rollback_conn, lambda ts: [[0.0] * 3 for _ in ts],
                               model_name="fake")


def test_sync_skips_grouped_research_sections(rollback_conn):
    sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    leaked = q(rollback_conn, """
        SELECT count(*) AS n FROM course_embeddings e
        WHERE NOT EXISTS (SELECT 1 FROM sections s
            WHERE s.group_id IS NULL AND s.subject_code = e.subject_code
              AND s.course_number = e.course_number
              AND COALESCE(s.title, '') = e.title)""")[0]["n"]
    assert leaked == 0


# --------------------------------------------------------------------------
# Search modes
# --------------------------------------------------------------------------

def test_stored_vectors_match_distinct_courses(conn, have_vectors):
    assert have_vectors == distinct_courses(conn)


def test_semantic_finds_paraphrase_without_shared_words(conn, have_vectors):
    top = courses(search_sections(conn, "lessons for a brass instrument",
                                  mode="semantic", limit=40))[:5]
    if not any(c.startswith("MUSI") for c, _ in top):
        pytest.skip("no music courses in the data")
    assert top[0][0].startswith("MUSI"), top
    keyword = courses(search_sections(
        conn, "lessons for a brass instrument", mode="keyword", limit=40))[:5]
    assert not any(c.startswith("MUSI") for c, _ in keyword), keyword


def test_semantic_finds_machine_learning_for_a_description(conn, have_vectors):
    got = courses(search_sections(
        conn, "teaching computers to learn patterns from data",
        mode="semantic", limit=100))[:10]
    if not q(conn, "SELECT 1 FROM sections WHERE title = 'MACHINE LEARNING'"):
        pytest.skip("no MACHINE LEARNING course in the data")
    assert ("COMP 642", "MACHINE LEARNING") in got


def test_semantic_scores_are_similarities_above_the_cutoff(conn, have_vectors):
    got = search_sections(conn, "writing poetry", mode="semantic", limit=100)
    assert got
    scores = [r.score for r in got]
    assert scores == sorted(scores, reverse=True)
    assert all(MIN_SIMILARITY <= s <= 1 for s in scores)
    # Sections of one course share a score.
    by_course = {}
    for r in got:
        by_course.setdefault((r.subject_code, r.course_number, r.title),
                             set()).add(r.score)
    assert all(len(v) == 1 for v in by_course.values())


def test_hybrid_ranks_both_signals_and_has_no_duplicates(conn, have_vectors):
    got = search_sections(conn, "learning to play piano", mode="hybrid",
                          limit=100)
    assert got
    assert len({r.crn for r in got}) == len(got)
    scores = [r.score for r in got]
    assert scores == sorted(scores, reverse=True)
    # The piano course is in both rankings, so it beats keyword-only noise
    # such as "machine learning".
    assert got[0].subject_code == "MUSI"


def test_hybrid_returns_keyword_matches_too(conn, have_vectors):
    keyword = {r.crn for r in search_sections(conn, "calculus", mode="keyword",
                                              limit=200)}
    hybrid = {r.crn for r in search_sections(conn, "calculus", mode="hybrid",
                                             limit=200)}
    assert keyword <= hybrid


def test_hybrid_respects_filters(conn, have_vectors):
    f = SectionFilters(subject="COMP", level_min=300, level_max=499)
    for mode in ("semantic", "hybrid"):
        got = search_sections(conn, "how to write code", f, mode=mode, limit=100)
        assert got, mode
        assert all(r.subject_code == "COMP"
                   and 300 <= int(r.course_number) <= 499 for r in got), mode
    open_only = search_sections(
        conn, "programming", SectionFilters(open_seats_only=True),
        mode="hybrid", limit=100)
    assert all(r.open_seats > 0 for r in open_only)


def test_filters_apply_before_ranking(conn, have_vectors):
    """A filter that excludes the best matches still yields full results."""
    f = SectionFilters(subject="ENGL")
    got = search_sections(conn, "calculus integration", f, mode="semantic",
                          limit=50, min_similarity=-1)
    assert got and all(r.subject_code == "ENGL" for r in got)


def test_nonsense_queries_return_nothing(conn, have_vectors):
    for text in ("zzzzqqqq", "quantum chromodynamics"):
        assert search_sections(conn, text, mode="hybrid") == [], text
        assert search_sections(conn, text, mode="semantic") == [], text


def test_course_code_still_ranks_first_in_every_mode(conn, have_vectors):
    for mode in ("auto", "keyword", "hybrid"):
        got = search_sections(conn, "COMP 140", mode=mode, limit=5)
        assert (got[0].subject_code, got[0].course_number) == ("COMP", "140"), mode


def test_auto_is_hybrid_when_vectors_exist(conn, have_vectors):
    text = "learning to play piano"
    auto = search_sections(conn, text, limit=30)
    hybrid = search_sections(conn, text, mode="hybrid", limit=30)
    assert [(r.crn, r.score) for r in auto] == [(r.crn, r.score) for r in hybrid]


def test_auto_falls_back_to_keyword_without_vectors(conn):
    kw = search_sections(conn, "calculus", mode="keyword", limit=30)
    auto = search_sections(conn, "calculus", limit=30, model_name="no-such-model",
                           embed_fn=lambda ts: pytest.fail("must not embed"))
    assert [(r.crn, r.score) for r in auto] == [(r.crn, r.score) for r in kw]


def test_explicit_semantic_or_hybrid_without_vectors_raises(conn):
    for mode in ("semantic", "hybrid"):
        with pytest.raises(RuntimeError):
            search_sections(conn, "calculus", mode=mode,
                            model_name="no-such-model", embed_fn=fake_embed)


def test_empty_query_never_embeds(conn):
    got = search_sections(conn, "", SectionFilters(subject="MATH"),
                          mode="hybrid", model_name="no-such-model",
                          embed_fn=lambda ts: pytest.fail("must not embed"))
    assert got


def test_invalid_mode_raises(conn):
    with pytest.raises(ValueError):
        search_sections(conn, "x", mode="fuzzy")


def test_custom_embedder_and_model_name(rollback_conn):
    """With every course embedded by a fake model and no similarity floor,
    semantic search returns exactly the sections the filters allow."""
    sync_course_embeddings(rollback_conn, fake_embed, model_name="fake")
    f = SectionFilters(subject="MATH")
    everything = search_sections(rollback_conn, None, f, limit=200)
    got = search_sections(rollback_conn, "anything", f, mode="semantic",
                          embed_fn=fake_embed, model_name="fake",
                          min_similarity=-1, limit=200)
    assert {r.crn for r in got} == {r.crn for r in everything}
    hybrid = search_sections(rollback_conn, "calculus", f, mode="hybrid",
                             embed_fn=fake_embed, model_name="fake",
                             min_similarity=-1, limit=200)
    assert {r.crn for r in hybrid} == {r.crn for r in everything}
