"""Tests for the service layer, run against the live Docker database.

Needs ``DATABASE_URL`` (and ``docker compose up -d``):

    export DATABASE_URL=postgresql://planner:planner@localhost:5433/rice_planner
    pytest

The data is real and changes with every crawl, so tests do not hard-code
CRNs. They pick rows from the database and derive the expected answer from
them, or compare a service function with an independent computation in
Python or plain SQL. Everything is read-only.
"""

import json
import os
import re
from datetime import date, datetime, time, timedelta

import psycopg
import pytest
from psycopg.rows import dict_row

from services import (BusyBlock, SectionFilters, check_schedule_conflicts,
                      find_sections_avoiding, get_course_sections,
                      get_section, list_distribution_groups, list_subjects,
                      search_sections, seat_status)
from services.models import DAY_LETTERS

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL is not set; point it at the Docker database")


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as c:
        yield c


def q(conn, sql, params=None):
    """Run plain SQL, return all rows as dicts."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@pytest.fixture(scope="module")
def term(conn):
    t = q(conn, "SELECT max(term_code) AS t FROM sections WHERE is_active")[0]["t"]
    assert t, "database has no active sections"
    return t


@pytest.fixture(scope="module")
def meeting(conn, term):
    """A real meeting of an ordinary active section, with its course info."""
    rows = q(conn, """
        SELECT m.crn, m.day_of_week, m.start_time, m.end_time,
               m.start_date, m.end_date, s.subject_code, s.course_number
        FROM meetings m JOIN sections s USING (term_code, crn)
        WHERE m.term_code = %s AND s.is_active AND s.group_id IS NULL
        ORDER BY m.crn, m.day_of_week LIMIT 1""", (term,))
    assert rows, "no meetings in the database"
    return rows[0]


def narrow(m):
    """Filters that select only the course of meeting ``m`` (small result)."""
    return SectionFilters(subject=m["subject_code"],
                          level_min=int(m["course_number"]),
                          level_max=int(m["course_number"]))


def hhmm(t):
    return t.strftime("%H:%M")


def block_for(m, **overrides):
    """A BusyBlock equal to meeting ``m`` unless overridden."""
    args = dict(day=DAY_LETTERS[m["day_of_week"]], start=hhmm(m["start_time"]),
                end=hhmm(m["end_time"]))
    args.update(overrides)
    return BusyBlock(**args)


def shift(t, minutes):
    return (datetime.combine(date(2000, 1, 1), t)
            + timedelta(minutes=minutes)).time()


def crns(results):
    return {r.crn for r in results}


# --------------------------------------------------------------------------
# BusyBlock validation
# --------------------------------------------------------------------------

def test_busy_block_accepts_tool_format():
    b = BusyBlock.from_dict({"day": "m", "start": "10:00", "end": "11:15"})
    assert (b.day, b.day_num) == ("M", 1)
    assert (b.start, b.end) == (time(10, 0), time(11, 15))
    assert BusyBlock("R", "9:30", "10:45").day_num == 4   # R = Thursday


@pytest.mark.parametrize("kwargs", [
    dict(day="X", start="10:00", end="11:00"),            # bad day letter
    dict(day="M", start="11:00", end="10:00"),            # end before start
    dict(day="M", start="10:00", end="10:00"),            # empty
    dict(day="M", start="10am", end="11:00"),             # bad time format
    dict(day="M", start="25:00", end="26:00"),            # impossible time
    dict(day="M", start="10:00", end="11:00",
         start_date="2026-12-01", end_date="2026-11-01"),  # dates reversed
    dict(day="M", start="10:00", end="11:00", start_date="12/01"),
])
def test_busy_block_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        BusyBlock(**kwargs)


# --------------------------------------------------------------------------
# find_sections_avoiding
# --------------------------------------------------------------------------

def test_avoiding_excludes_section_in_the_blocked_slot(conn, meeting):
    f = narrow(meeting)
    assert meeting["crn"] in crns(find_sections_avoiding(conn, [], f, limit=200))
    got = find_sections_avoiding(conn, [block_for(meeting)], f, limit=200)
    assert meeting["crn"] not in crns(got)


def test_avoiding_partial_overlap_counts(conn, meeting):
    f = narrow(meeting)
    start = meeting["start_time"]
    # A one-minute block in the first minute of the meeting.
    block = block_for(meeting, end=hhmm(shift(start, 1)))
    assert meeting["crn"] not in crns(
        find_sections_avoiding(conn, [block], f, limit=200))


def test_avoiding_back_to_back_is_not_a_conflict(conn, meeting):
    f = narrow(meeting)
    before = block_for(meeting, start=hhmm(shift(meeting["start_time"], -60)),
                       end=hhmm(meeting["start_time"]))
    after = block_for(meeting, start=hhmm(meeting["end_time"]),
                      end=hhmm(shift(meeting["end_time"], 60)))
    got = crns(find_sections_avoiding(conn, [before, after], f, limit=200))
    assert meeting["crn"] in got


def test_avoiding_other_day_is_not_a_conflict(conn, meeting):
    other = DAY_LETTERS[meeting["day_of_week"] % 7 + 1]
    # Only safe if the section does not also meet on that other day.
    days = {r["day_of_week"] for r in q(
        conn, "SELECT day_of_week FROM meetings WHERE crn = %s",
        (meeting["crn"],))}
    if meeting["day_of_week"] % 7 + 1 in days:
        pytest.skip("sample section meets on the neighbouring day too")
    block = block_for(meeting, day=other)
    assert meeting["crn"] in crns(
        find_sections_avoiding(conn, [block], narrow(meeting), limit=200))


def test_avoiding_respects_date_ranges(conn, meeting):
    f = narrow(meeting)
    start, end = meeting["start_date"], meeting["end_date"]
    # Block active only before the meeting's first day: no conflict.
    early = block_for(meeting, start_date=date(2000, 1, 1),
                      end_date=start - timedelta(days=1))
    assert meeting["crn"] in crns(
        find_sections_avoiding(conn, [early], f, limit=200))
    # Block active only after its last day: no conflict.
    late = block_for(meeting, start_date=end + timedelta(days=1),
                     end_date=end + timedelta(days=30))
    assert meeting["crn"] in crns(
        find_sections_avoiding(conn, [late], f, limit=200))
    # Block covering the meeting's dates: conflict.
    during = block_for(meeting, start_date=start, end_date=end)
    assert meeting["crn"] not in crns(
        find_sections_avoiding(conn, [during], f, limit=200))
    # A block touching only the last day still conflicts (inclusive range).
    last_day = block_for(meeting, start_date=end, end_date=end)
    assert meeting["crn"] not in crns(
        find_sections_avoiding(conn, [last_day], f, limit=200))


def test_avoiding_matches_python_brute_force(conn, term):
    """Compare the SQL answer with a plain-Python overlap computation."""
    subject = "MATH"
    blocks = [
        BusyBlock("M", "09:00", "10:30"), BusyBlock("W", "09:00", "10:30"),
        BusyBlock("T", "13:00", "14:15"), BusyBlock("R", "11:00", "12:00"),
        BusyBlock("F", "08:00", "17:00"),
    ]
    f = SectionFilters(subject=subject)
    all_ = search_sections(conn, None, f, limit=200)
    meetings = q(conn, "SELECT crn, day_of_week, start_time, end_time, "
                 "start_date, end_date FROM meetings WHERE term_code = %s",
                 (term,))

    def clashes(m, b):
        return (m["day_of_week"] == b.day_num
                and m["start_time"] < b.end and b.start < m["end_time"])

    bad = {m["crn"] for m in meetings for b in blocks if clashes(m, b)}
    expected = {s.crn for s in all_} - bad
    got = crns(find_sections_avoiding(conn, blocks, f, limit=200))
    assert got == expected
    assert bad & {s.crn for s in all_}, "test blocks excluded nothing"


def test_avoiding_composes_with_filters(conn, term):
    blocks = [BusyBlock("M", "09:00", "10:30")]
    f = SectionFilters(subject="COMP", level_min=100, level_max=299,
                       open_seats_only=True)
    got = find_sections_avoiding(conn, blocks, f, limit=200)
    for s in got:
        assert s.subject_code == "COMP"
        assert 100 <= int(s.course_number) <= 299
        assert s.open_seats and s.open_seats > 0


def test_avoiding_includes_sections_without_meetings(conn, term):
    no_meet = q(conn, """
        SELECT s.crn FROM sections s
        WHERE s.term_code = %s AND s.is_active AND s.group_id IS NULL
          AND NOT EXISTS (SELECT 1 FROM meetings m
                          WHERE m.term_code = s.term_code AND m.crn = s.crn)
        LIMIT 1""", (term,))
    if not no_meet:
        pytest.skip("every ordinary section has meetings")
    s = get_section(conn, no_meet[0]["crn"])
    blocks = [BusyBlock(d, "00:00", "23:59") for d in "MTWRFSU"]
    got = find_sections_avoiding(
        conn, blocks, SectionFilters(subject=s.subject_code), limit=200)
    assert no_meet[0]["crn"] in crns(got)


def test_avoiding_rejects_invalid_block(conn):
    with pytest.raises(ValueError):
        find_sections_avoiding(conn, [{"day": "Z", "start": "10:00",
                                       "end": "11:00"}])


# --------------------------------------------------------------------------
# check_schedule_conflicts
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def clashing_pair(conn, term):
    rows = q(conn, """
        SELECT a.crn AS crn_a, b.crn AS crn_b, a.day_of_week AS day,
               greatest(a.start_time, b.start_time) AS lo,
               least(a.end_time, b.end_time) AS hi
        FROM meetings a JOIN meetings b
          ON a.term_code = b.term_code AND a.day_of_week = b.day_of_week
         AND a.crn < b.crn
        WHERE a.term_code = %s
          AND a.start_time < b.end_time AND b.start_time < a.end_time
          AND a.start_date <= b.end_date AND b.start_date <= a.end_date
        ORDER BY a.crn, b.crn LIMIT 1""", (term,))
    if not rows:
        pytest.skip("no overlapping sections in the data")
    return rows[0]


def test_conflicts_reports_overlapping_pair(conn, clashing_pair):
    p = clashing_pair
    r = check_schedule_conflicts(conn, [p["crn_b"], p["crn_a"]])
    assert not r.ok and not r.unknown_crns
    match = [c for c in r.conflicts
             if (c.crn_a, c.crn_b, c.day) == (p["crn_a"], p["crn_b"], p["day"])]
    assert match
    assert match[0].overlap_start == hhmm(p["lo"])
    assert match[0].overlap_end == hhmm(p["hi"])
    assert match[0].day_letter == DAY_LETTERS[p["day"]]


def test_conflicts_single_and_duplicate_crn_is_clean(conn, meeting):
    c = meeting["crn"]
    for given in ([c], [c, c]):
        r = check_schedule_conflicts(conn, given)
        assert r.ok and r.conflicts == [] and r.unknown_crns == []


def test_conflicts_back_to_back_sections_do_not_conflict(conn, term):
    rows = q(conn, """
        SELECT a.crn AS crn_a, b.crn AS crn_b
        FROM meetings a JOIN meetings b
          ON a.term_code = b.term_code AND a.day_of_week = b.day_of_week
         AND a.end_time = b.start_time AND a.crn <> b.crn
        WHERE a.term_code = %s
          AND NOT EXISTS (   -- the pair has no other overlapping meetings
            SELECT 1 FROM meetings x JOIN meetings y
              ON x.term_code = y.term_code AND x.day_of_week = y.day_of_week
            WHERE x.term_code = a.term_code AND x.crn = a.crn AND y.crn = b.crn
              AND x.start_time < y.end_time AND y.start_time < x.end_time)
        LIMIT 1""", (term,))
    if not rows:
        pytest.skip("no back-to-back pair in the data")
    r = check_schedule_conflicts(conn, [rows[0]["crn_a"], rows[0]["crn_b"]])
    assert r.conflicts == []


def test_conflicts_reports_unknown_crns(conn, meeting):
    r = check_schedule_conflicts(conn, [meeting["crn"], "00000"])
    assert r.unknown_crns == ["00000"]
    assert not r.ok
    assert check_schedule_conflicts(conn, []).ok


def test_conflicts_agree_with_python_brute_force(conn, term):
    crn_list = [r["crn"] for r in q(conn, """
        SELECT DISTINCT crn FROM meetings WHERE term_code = %s
        ORDER BY crn LIMIT 40""", (term,))]
    meets = q(conn, "SELECT crn, day_of_week, start_time, end_time, "
              "start_date, end_date FROM meetings WHERE term_code = %s "
              "AND crn = ANY(%s)", (term, crn_list))
    expected = set()
    for a in meets:
        for b in meets:
            if (a["crn"] < b["crn"] and a["day_of_week"] == b["day_of_week"]
                    and a["start_time"] < b["end_time"]
                    and b["start_time"] < a["end_time"]
                    and a["start_date"] <= b["end_date"]
                    and b["start_date"] <= a["end_date"]):
                expected.add((a["crn"], b["crn"], a["day_of_week"],
                              max(a["start_time"], b["start_time"]),
                              min(a["end_time"], b["end_time"])))
    r = check_schedule_conflicts(conn, crn_list)
    got = {(c.crn_a, c.crn_b, c.day, time.fromisoformat(c.overlap_start),
            time.fromisoformat(c.overlap_end)) for c in r.conflicts}
    assert got == expected


# --------------------------------------------------------------------------
# search_sections and filters
# --------------------------------------------------------------------------

def test_search_empty_query_lists_by_filters(conn):
    got = search_sections(conn, None, SectionFilters(subject="COMP"), limit=10)
    assert 0 < len(got) <= 10
    assert all(s.subject_code == "COMP" for s in got)


def test_search_course_code_ranks_course_first(conn, meeting):
    code = f"{meeting['subject_code']} {meeting['course_number']}"
    for text in (code, code.lower(), code.replace(" ", "")):
        got = search_sections(conn, text, limit=5)
        assert got, text
        assert (got[0].subject_code, got[0].course_number) == (
            meeting["subject_code"], meeting["course_number"]), text


def test_search_finds_a_section_by_title_keyword(conn, meeting):
    title = get_section(conn, meeting["crn"]).title
    words = [w for w in re.findall(r"[A-Za-z]{5,}", title or "")]
    if not words:
        pytest.skip("sample title has no long word")
    got = search_sections(conn, words[0], limit=200)
    assert meeting["crn"] in crns(got)
    scores = [s.score for s in got]
    assert scores == sorted(scores, reverse=True)


def test_search_description_words_and_no_results(conn):
    assert search_sections(conn, "zzzzqqqqxxxx") == []


def test_search_is_safe_with_special_characters(conn):
    for text in ("100%", "a_b", "O'Brien; DROP TABLE sections", "\\", "()",
                 "c++ & friends"):
        search_sections(conn, text)   # must not raise
    assert q(conn, "SELECT count(*) AS n FROM sections")[0]["n"] > 0


def test_filter_subject_accepts_one_or_many(conn):
    one = search_sections(conn, None, SectionFilters(subject="math"), limit=200)
    assert one and {s.subject_code for s in one} == {"MATH"}
    many = search_sections(
        conn, None, SectionFilters(subject=["MATH", "COMP"]), limit=200)
    assert {s.subject_code for s in many} == {"MATH", "COMP"}


def test_filter_level_range(conn):
    got = search_sections(conn, None, SectionFilters(
        subject="COMP", level_min=300, level_max=399), limit=200)
    assert got
    assert all(300 <= int(s.course_number) <= 399 for s in got)
    open_ended = search_sections(
        conn, None, SectionFilters(subject="COMP", level_min=400), limit=200)
    assert all(int(s.course_number) >= 400 for s in open_ended)


def test_filter_open_seats(conn):
    got = search_sections(conn, None, SectionFilters(open_seats_only=True),
                          limit=200)
    assert got and all(s.open_seats > 0 for s in got)
    closed = q(conn, """SELECT count(*) AS n FROM sections
                        WHERE is_active AND group_id IS NULL
                          AND max_enrollment - enrolled <= 0""")[0]["n"]
    everything = search_sections(conn, None, limit=200)
    if closed and len(everything) < 200:
        assert len(got) < len(everything)


def test_filter_method_is_case_insensitive(conn):
    got = search_sections(conn, None, SectionFilters(method="online"), limit=200)
    assert got and all(s.method == "Online" for s in got)


def test_filter_distribution_group_full_and_short_names(conn):
    groups = list_distribution_groups(conn)
    assert groups
    name = groups[0].name
    full = search_sections(
        conn, None, SectionFilters(distribution_group=name), limit=200)
    assert full and all(s.distribution_group == name for s in full)
    m = re.fullmatch(r"Distribution Group (I+)", name)
    if m:
        short = search_sections(
            conn, None, SectionFilters(distribution_group=m.group(1)),
            limit=200)
        assert crns(short) == crns(full)


def _credit_numbers(text):
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text or "")]


def test_filter_credit_range(conn):
    got = search_sections(conn, None, SectionFilters(
        credits_min=3, credits_max=3), limit=200)
    assert got
    for s in got:
        nums = _credit_numbers(s.credits)
        assert min(nums) <= 3 <= max(nums), s.credits
    # A 1-4 credit variable section counts for "3 credits"; a plain 1 does not.
    credits_seen = {s.credits for s in got}
    assert "3" in credits_seen
    assert "1" not in credits_seen and "4" not in credits_seen
    none = search_sections(conn, None, SectionFilters(
        credits_min=100, credits_max=200))
    assert none == []


def test_filter_instructor_by_name_and_netid(conn, term):
    row = q(conn, """
        SELECT i.netid, i.name, si.crn FROM section_instructors si
        JOIN instructors i ON i.key = si.instructor_key
        JOIN sections s USING (term_code, crn)
        WHERE si.term_code = %s AND s.is_active AND s.group_id IS NULL
          AND i.netid IS NOT NULL LIMIT 1""", (term,))
    if not row:
        pytest.skip("no instructors with a netid")
    row = row[0]
    by_net = search_sections(
        conn, None, SectionFilters(instructor=row["netid"]), limit=200)
    assert row["crn"] in crns(by_net)
    surname = row["name"].split(",")[0].split()[-1]
    by_name = search_sections(
        conn, None, SectionFilters(instructor=surname.upper()), limit=200)
    assert row["crn"] in crns(by_name)
    for s in by_net:
        assert any(row["name"] == n for n in s.instructors)


def test_instructor_filter_treats_wildcards_literally(conn):
    assert search_sections(conn, None, SectionFilters(instructor="%%%")) == []


def test_grouped_research_sections_hidden_by_default(conn):
    grouped = q(conn, "SELECT crn FROM sections WHERE group_id IS NOT NULL "
                "AND is_active LIMIT 1")
    if not grouped:
        pytest.skip("no grouped sections")
    crn = grouped[0]["crn"]
    s = get_section(conn, crn)
    f = dict(subject=s.subject_code)
    default = search_sections(conn, None, SectionFilters(**f), limit=200)
    assert crn not in crns(default)
    with_group = search_sections(
        conn, None, SectionFilters(include_grouped=True, **f), limit=200)
    assert len(with_group) > len(default)


def test_summary_fields_are_consistent(conn):
    for s in search_sections(conn, "calculus", limit=10):
        d = get_section(conn, s.crn)
        assert (d.subject_code, d.course_number, d.title) == (
            s.subject_code, s.course_number, s.title)
        assert d.open_seats == s.open_seats
        assert sorted(i.name for i in d.instructors) == s.instructors


def test_unknown_term_returns_nothing(conn):
    f = SectionFilters(term_code="000000")
    assert search_sections(conn, "calculus", f) == []
    assert find_sections_avoiding(conn, [], f) == []


# --------------------------------------------------------------------------
# get_section / get_course_sections / seat_status
# --------------------------------------------------------------------------

def test_get_section_matches_database(conn, term, meeting):
    d = get_section(conn, meeting["crn"])
    row = q(conn, "SELECT * FROM sections WHERE term_code = %s AND crn = %s",
            (term, meeting["crn"]))[0]
    assert d.term_code == term
    assert (d.subject_code, d.title, d.credits) == (
        row["subject_code"], row["title"], row["credits"])
    assert d.prerequisites == row["prerequisites"]
    n_meet = q(conn, "SELECT count(*) AS n FROM meetings WHERE term_code = %s "
               "AND crn = %s", (term, meeting["crn"]))[0]["n"]
    assert len(d.meetings) == n_meet
    assert all(m.day_letter == DAY_LETTERS[m.day] for m in d.meetings)
    n_inst = q(conn, "SELECT count(*) AS n FROM section_instructors "
               "WHERE term_code = %s AND crn = %s", (term, meeting["crn"]))[0]["n"]
    assert len(d.instructors) == n_inst


def test_get_section_unknown_returns_none(conn):
    assert get_section(conn, "00000") is None
    assert get_section(conn, "00000", term_code="000000") is None


def test_get_section_includes_multiple_instructors_and_meetings(conn, term):
    multi = q(conn, """SELECT crn FROM section_instructors WHERE term_code = %s
                       GROUP BY crn HAVING count(*) > 1 LIMIT 1""", (term,))
    if multi:
        assert len(get_section(conn, multi[0]["crn"]).instructors) > 1
    multi = q(conn, """SELECT crn FROM meetings WHERE term_code = %s
                       GROUP BY crn HAVING count(*) > 2 LIMIT 1""", (term,))
    if multi:
        assert len(get_section(conn, multi[0]["crn"]).meetings) > 2


def test_get_course_sections_returns_every_section(conn, term, meeting):
    got = get_course_sections(
        conn, meeting["subject_code"].lower(), meeting["course_number"])
    expected = q(conn, """SELECT crn FROM sections WHERE term_code = %s
        AND subject_code = %s AND course_number = %s AND is_active""",
                 (term, meeting["subject_code"], meeting["course_number"]))
    assert {d.crn for d in got} == {r["crn"] for r in expected}
    assert meeting["crn"] in {d.crn for d in got}
    assert get_course_sections(conn, "COMP", "999") == []


def test_seat_status_matches_database(conn, term, meeting):
    r = seat_status(conn, [meeting["crn"], "00000", meeting["crn"]])
    assert [s.crn for s in r] == [meeting["crn"], "00000"]   # deduped, ordered
    ok, missing = r
    assert ok.found and not missing.found
    row = q(conn, "SELECT * FROM sections WHERE term_code = %s AND crn = %s",
            (term, meeting["crn"]))[0]
    assert (ok.enrolled, ok.max_enrollment, ok.waitlisted) == (
        row["enrolled"], row["max_enrollment"], row["waitlisted"])
    assert ok.open_seats == max(0, row["max_enrollment"] - row["enrolled"])
    assert ok.scraped_at == row["scraped_at"]
    assert ok.enrollment_as_of == row["enrollment_as_of"]
    assert seat_status(conn, []) == []


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

def test_list_subjects(conn, term):
    subjects = list_subjects(conn)
    assert subjects
    codes = [s.code for s in subjects]
    assert codes == sorted(codes)
    by_code = {s.code: s for s in subjects}
    total = q(conn, "SELECT count(*) AS n FROM sections WHERE is_active "
              "AND term_code = %s", (term,))[0]["n"]
    assert sum(s.active_sections for s in subjects) == total
    assert by_code["COMP"].active_sections > 0
    assert by_code["COMP"].name


def test_list_distribution_groups(conn, term):
    groups = list_distribution_groups(conn)
    assert groups
    expected = {r["distribution_group"]: r["n"] for r in q(conn, """
        SELECT distribution_group, count(*)::int AS n FROM sections
        WHERE is_active AND term_code = %s AND distribution_group IS NOT NULL
        GROUP BY 1""", (term,))}
    assert {g.name: g.active_sections for g in groups} == expected


# --------------------------------------------------------------------------
# JSON output
# --------------------------------------------------------------------------

def test_results_serialize_to_json(conn, meeting):
    d = get_section(conn, meeting["crn"])
    json.dumps(d.to_dict())
    json.dumps([s.to_dict() for s in search_sections(conn, "calculus")])
    json.dumps(check_schedule_conflicts(conn, [meeting["crn"], "0"]).to_dict())
    json.dumps([s.to_dict() for s in seat_status(conn, [meeting["crn"]])])
    json.dumps([s.to_dict() for s in list_subjects(conn)])
    assert d.to_dict()["meetings"][0]["start_time"] == hhmm(
        meeting["start_time"]) or len(d.meetings) > 1
