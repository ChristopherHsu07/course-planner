"""Tests for the restriction-text parser, using real strings from Rice."""

import os

import pytest

from agent.restrictions import parse_restrictions

UG = ("Must be enrolled in one of the following Level(s):; "
      "Undergraduate Professional; Visiting Undergraduate; Undergraduate")
GRAD = "Must be enrolled in one of the following Level(s):; Graduate"


def test_undergraduate_only():
    r = parse_restrictions(UG)
    assert r.parsed
    assert r.requirements == {"Level": ["Undergraduate Professional",
                                        "Visiting Undergraduate",
                                        "Undergraduate"]}
    assert r.summary == "Undergraduate students only"


def test_graduate_only():
    assert parse_restrictions(GRAD).summary == "Graduate students only"


def test_level_plus_major_and_class():
    r = parse_restrictions(
        UG + "; Must be enrolled in one of the following Major(s):; English; "
        "Must be enrolled in one of the following Class(es):; Senior")
    assert r.requirements["Major"] == ["English"]
    assert r.requirements["Class"] == ["Senior"]        # "(es)" header
    assert r.summary == ("Undergraduate students only; Major: English; "
                         "Class: Senior")


def test_programs_and_level():
    r = parse_restrictions(
        "Must be enrolled in one of the following Program(s):; "
        "Online Master of Data Science; Online Master Computer Science; "
        "Must be enrolled in one of the following Level(s):; Graduate")
    assert r.summary == ("Program: Online Master of Data Science, Online "
                         "Master Computer Science; Graduate students only")


def test_cohort_list_is_truncated():
    cohorts = [f"Cohort {i}" for i in range(8)]
    r = parse_restrictions(
        GRAD + "; Must belong to one of the following Cohort(s):; "
        + "; ".join(cohorts))
    assert r.requirements["Cohort"] == cohorts
    assert "and 4 more" in r.summary
    assert "Cohort 7" not in r.summary


def test_excluded_form():
    r = parse_restrictions(
        "Must not be enrolled in one of the following Major(s):; Music")
    assert r.excluded == {"Major": ["Music"]}
    assert r.summary == "Not allowed: Major: Music"


def test_no_restrictions():
    for raw in (None, "", "   "):
        r = parse_restrictions(raw)
        assert r.summary is None and r.requirements == {}


def test_unrecognised_text_falls_back_to_raw():
    r = parse_restrictions("Instructor approval and a portfolio review")
    assert not r.parsed
    assert r.summary == "Instructor approval and a portfolio review"
    long = parse_restrictions("x " * 300)
    assert len(long.summary) <= 200 and long.summary.endswith("...")


@pytest.mark.skipif(not os.environ.get("DATABASE_URL"),
                    reason="DATABASE_URL is not set")
def test_every_restriction_in_the_database_parses():
    import psycopg
    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        rows = c.execute("SELECT DISTINCT restrictions FROM sections "
                         "WHERE restrictions IS NOT NULL").fetchall()
    assert rows
    for (raw,) in rows:
        r = parse_restrictions(raw)
        assert r.parsed, raw
        assert r.summary and len(r.summary) < len(raw) + 40, raw
