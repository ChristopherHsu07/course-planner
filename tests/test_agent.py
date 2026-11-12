"""Tests for the course assistant, run with a scripted fake model.

No Ollama is needed: ``FakeLLM`` plays back prepared replies, so these tests
check everything the code (not the model) is responsible for: the tool loop,
busy-block handling, grounding, citations, flags, the freshness footer. They
run against the live Docker database (``DATABASE_URL``), read-only.

One optional test (marked ``ollama``) talks to a real local model; it is
skipped unless Ollama is running with the default model pulled.
"""

import copy
import json
import os
from datetime import datetime

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

from agent import config, guard
from agent.core import FALLBACK_REPLY, run_turn
from agent.llm import LLMError, OllamaClient
from agent.prompts import build_system_prompt
from agent.schedule import extract_busy_blocks, looks_like_schedule
from agent.session import Session
from agent.tools import (NO_MEETING, dispatch, record_seen,
                         resolve_subject_text, tool_definitions)
from agent.util import format_as_of
from services import BusyBlock, SectionFilters, search_sections
from services.models import Subject

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL is not set; point it at the Docker database")


# --------------------------------------------------------------------------
# Fakes and helpers
# --------------------------------------------------------------------------

def say(text):
    return {"role": "assistant", "content": text}


def tool_call(name, **arguments):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name,
                                         "arguments": arguments}}]}


def extraction(blocks=(), clarification=""):
    return {"role": "assistant", "content": json.dumps(
        {"busy_blocks": list(blocks), "clarification": clarification})}


def last_tool_result(messages):
    tool_msgs = [m for m in messages if m.get("role") == "tool"]
    return json.loads(tool_msgs[-1]["content"])


def first_crn(messages):
    return last_tool_result(messages)["results"][0]["crn"]


class FakeLLM:
    """Plays back prepared replies.

    ``main`` answers chat calls; ``extraction`` answers calls that carry a
    JSON-schema ``format`` (the schedule extractor). An item may be a dict or
    a function of the messages so far. Every call is recorded in ``calls``.
    """

    def __init__(self, main=(), extraction=()):
        self.main = list(main)
        self.extraction = list(extraction)
        self.calls = []

    def chat(self, messages, tools=None, format=None, think=None):
        self.calls.append({"messages": copy.deepcopy(list(messages)),
                           "tools": tools, "format": format, "think": think})
        queue = self.extraction if format is not None else self.main
        assert queue, "the agent made an unexpected model call"
        item = queue.pop(0)
        return item(messages) if callable(item) else item

    @property
    def main_calls(self):
        return [c for c in self.calls if c["format"] is None]

    @property
    def extraction_calls(self):
        return [c for c in self.calls if c["format"] is not None]


def q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


@pytest.fixture(scope="module")
def conn():
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row) as c:
        yield c


@pytest.fixture
def session():
    return Session()


@pytest.fixture(autouse=True)
def keyword_search_only(monkeypatch):
    """Agent tests check the agent, not search ranking.

    Loading the embedding model takes minutes the first time in a process, so
    keep it out of these tests (semantic search has its own tests).
    """
    monkeypatch.setattr(config, "SEMANTIC", False)


@pytest.fixture(scope="module")
def comp_meeting(conn):
    """A meeting of an ordinary COMP section (to build a real busy block)."""
    rows = q(conn, """
        SELECT m.crn, m.day_of_week, m.start_time, m.end_time,
               s.subject_code, s.course_number
        FROM meetings m JOIN sections s USING (term_code, crn)
        WHERE s.is_active AND s.group_id IS NULL AND s.subject_code = 'COMP'
        ORDER BY m.crn, m.day_of_week LIMIT 1""")
    assert rows
    return rows[0]


def block_dict(m):
    return {"day": "MTWRFSU"[m["day_of_week"] - 1],
            "start": m["start_time"].strftime("%H:%M"),
            "end": m["end_time"].strftime("%H:%M")}


def crns_of(envelope):
    return [r["crn"] for r in envelope["results"]]


# --------------------------------------------------------------------------
# The tool loop and the final answer
# --------------------------------------------------------------------------

def test_tool_loop_runs_tool_then_answers_with_full_citation(conn, session):
    llm = FakeLLM(main=[
        tool_call("search_sections", query="calculus", subject="MATH"),
        lambda m: say(f"Try [CRN {first_crn(m)}] for calculus."),
    ])
    r = run_turn(session, "show me calculus", llm, conn)
    assert [c.name for c in r.tool_calls] == ["search_sections"]
    assert r.steps == 2 and not r.hit_step_limit
    crn = r.tool_calls[0].result["results"][0]["crn"]
    sec = session.seen[crn]
    # [CRN n] became subject, number, section and CRN.
    assert f"{sec['subject']} {sec['number']} section {sec['section']} " \
           f"(CRN {crn})" in r.reply
    assert "[CRN" not in r.reply
    assert r.cited_crns == [crn]
    # The tools were offered and the model got the tool result.
    assert llm.main_calls[0]["tools"] == tool_definitions()
    assert any(m["role"] == "tool" for m in llm.main_calls[1]["messages"])


def test_system_prompt_states_term_rules_and_busy_times(conn):
    text = build_system_prompt("202720", [BusyBlock("M", "10:00", "11:15")])
    assert "Spring 2027" in text and "202720" in text
    assert "M 10:00-11:15" in text
    for phrase in ("ONLY from tool results", "[CRN 12345]",
                   "check_schedule_conflicts", "NO SCHEDULED MEETING TIME",
                   "permission"):
        assert phrase in text
    assert "none" in build_system_prompt("202720", [])


def test_term_is_pinned_to_the_latest_loaded_term(conn, session):
    llm = FakeLLM(main=[say("Hello! What would you like to take?")])
    run_turn(session, "hi there", llm, conn)
    assert session.term_code == q(conn, "SELECT max(term_code) AS t FROM "
                                  "sections WHERE is_active")[0]["t"]
    assert "Spring 2027" in llm.main_calls[0]["messages"][0]["content"]


def test_answer_without_tools_has_no_footer(conn, session):
    r = run_turn(session, "hi there", FakeLLM(main=[say("Hello!")]), conn)
    assert r.reply == "Hello!"


def test_empty_model_answer_falls_back(conn, session):
    r = run_turn(session, "hi there", FakeLLM(main=[say("")]), conn)
    assert r.reply == FALLBACK_REPLY


# --------------------------------------------------------------------------
# Footer: freshness and the registration reminder
# --------------------------------------------------------------------------

def test_footer_has_data_timestamp_and_registrar_reminder(conn, session):
    llm = FakeLLM(main=[
        tool_call("search_sections", query="poetry", subject="ENGL"),
        lambda m: say(f"Consider [CRN {first_crn(m)}]."),
    ])
    r = run_turn(session, "poetry classes", llm, conn)
    crn = r.cited_crns[0]
    expected = format_as_of(datetime.fromisoformat(
        session.seen[crn]["enrollment_as_of"]))
    assert f"Data as of {expected} (Spring 2027)." in r.reply
    assert "ESTHER and the Registrar are the authoritative sources" in r.reply
    assert r.reply.rstrip().endswith("before you enroll.")


def test_footer_uses_oldest_timestamp_of_cited_sections(session):
    for crn, ts in (("11111", "2026-11-07T22:40:00"),
                    ("22222", "2026-11-07T15:42:00")):
        record_seen(session, term_code="202720", crn=crn, subject="COMP",
                    number="140", section="001", title="T",
                    permission_required=False, restrictions=None,
                    has_meetings=True,
                    enrollment_as_of=datetime.fromisoformat(ts))
    text = guard.footer(session, ["11111", "22222"], None)
    assert text.startswith("Data as of Nov 7, 2026 3:42 PM CST")


def test_footer_falls_back_to_term_freshness(conn):
    from services import get_data_freshness
    fresh = get_data_freshness(conn)
    text = guard.footer(Session(), [], fresh)
    assert format_as_of(fresh.enrollment_as_of_min) in text
    assert "(Spring 2027)" in text
    assert "unavailable" in guard.footer(Session(), [], None)


def test_format_as_of_labels_central_time_and_dst():
    assert format_as_of(datetime(2026, 11, 7, 15, 42)) == \
        "Nov 7, 2026 3:42 PM CST"
    assert format_as_of(datetime(2026, 7, 1, 0, 5)) == "Jul 1, 2026 12:05 AM CDT"
    assert format_as_of(None) is None


# --------------------------------------------------------------------------
# Busy blocks: parsed first, applied by code, never by the model
# --------------------------------------------------------------------------

def test_extracted_blocks_are_stored_and_applied_to_search(
        conn, session, comp_meeting):
    block = block_dict(comp_meeting)
    code = f"COMP {comp_meeting['course_number']}"
    msg = f"I have class on {block['day']} {block['start']} to {block['end']}"

    baseline = dispatch(conn, Session(), "search_sections",
                        {"query": code, "subject": "COMP", "limit": 15})
    assert comp_meeting["crn"] in crns_of(baseline)

    llm = FakeLLM(
        extraction=[extraction([block])],
        main=[tool_call("search_sections", query=code, subject="COMP",
                        limit=15),
              say("Done.")])
    r = run_turn(session, msg, llm, conn)
    assert session.busy_blocks == [BusyBlock.from_dict(block)]
    result = r.tool_calls[0].result
    assert result["busy_times_applied"] is True
    assert comp_meeting["crn"] not in crns_of(result)
    assert any("busy time" in n for n in result["notes"])
    # Same answer as calling the service directly with those blocks.
    direct = search_sections(
        conn, code, SectionFilters(subject="COMP"), limit=15,
        busy_blocks=session.busy_blocks)
    assert crns_of(result) == [s.crn for s in direct]


def test_model_cannot_override_busy_blocks(conn, session, comp_meeting):
    block = block_dict(comp_meeting)
    session.busy_blocks = [BusyBlock.from_dict(block)]
    code = f"COMP {comp_meeting['course_number']}"
    sneaky = dispatch(conn, session, "search_sections", {
        "query": code, "subject": "COMP", "limit": 15,
        "busy_blocks": [], "ignore_busy_times": True})
    assert comp_meeting["crn"] not in crns_of(sneaky)
    assert sneaky["ignored_arguments"] == ["busy_blocks", "ignore_busy_times"]
    assert "busy_blocks" not in json.dumps(tool_definitions())


def test_blocks_updates_accumulate_across_turns(conn, session):
    first = [{"day": "M", "start": "10:00", "end": "11:15"}]
    both = first + [{"day": "W", "start": "10:00", "end": "11:15"}]
    llm = FakeLLM(extraction=[extraction(first), extraction(both[::-1])],
                  main=[say("Noted."), say("Noted.")])
    run_turn(session, "I have class Monday 10:00 to 11:15", llm, conn)
    assert [b.day for b in session.busy_blocks] == ["M"]
    run_turn(session, "also class Wednesday same time", llm, conn)
    assert [b.day for b in session.busy_blocks] == ["M", "W"]   # sorted
    # The extractor was shown the blocks already on file.
    shown = llm.extraction_calls[1]["messages"][-1]["content"]
    assert '"day": "M"' in shown
    assert llm.extraction_calls[0]["think"] is False
    # The main model's system prompt lists them.
    assert "M 10:00-11:15, W 10:00-11:15" in \
        llm.main_calls[-1]["messages"][0]["content"]


def test_ambiguous_time_asks_a_question_and_stops(conn, session):
    llm = FakeLLM(extraction=[extraction(
        clarification="Is that 10 AM to 11 AM, or 10 PM to 11 PM?")])
    r = run_turn(session, "I work Fridays from 10 to 11", llm, conn)
    assert r.asked_clarification
    assert r.reply == "Is that 10 AM to 11 AM, or 10 PM to 11 PM?"
    assert r.tool_calls == [] and llm.main_calls == []   # nothing ran on a guess
    assert session.busy_blocks == []
    assert "Data as of" not in r.reply
    # The exchange is kept so the student's answer has context.
    assert [m["role"] for m in session.messages] == \
        ["system", "user", "assistant"]


def test_invalid_or_unreadable_extraction_becomes_a_question(conn):
    bad_block = extraction([{"day": "M", "start": "11:00", "end": "10:00"}])
    garbage = {"role": "assistant", "content": "not json"}
    wrong_shape = {"role": "assistant", "content": '{"busy_blocks": 5}'}
    for reply in (bad_block, garbage, wrong_shape):
        s = Session(busy_blocks=[BusyBlock("T", "09:00", "10:00")])
        update = extract_busy_blocks(FakeLLM(extraction=[reply]),
                                     s.busy_blocks, "class Monday 10 to 11am")
        assert update.clarification and "exact start and end times" in \
            update.clarification
        assert update.blocks == s.busy_blocks          # unchanged


def test_extraction_is_skipped_when_message_has_no_schedule(conn, session):
    llm = FakeLLM(main=[say("OK")])
    run_turn(session, "what is topology about", llm, conn)
    assert llm.extraction_calls == []


def test_class_request_does_not_become_busy_blocks(conn, session):
    """The 'MWF mornings' bug: a wish for class times is not a busy time."""
    llm = FakeLLM(main=[tool_call("search_sections", days="MWF",
                                  end_before="12:00"), say("Here you go.")])
    r = run_turn(session, "give me an analyzing diversity class that meets "
                 "in the mornings, monday wednesday and friday", llm, conn)
    assert llm.extraction_calls == []
    assert not r.asked_clarification and session.busy_blocks == []
    assert r.tool_calls[0].result["busy_times_applied"] is False


def test_prompt_tells_the_model_wanted_times_are_filters(conn):
    prompt = " ".join(build_system_prompt("202720", []).split())
    assert "time_of_day" in prompt and "not busy times" in prompt
    # It says which bound is which, so "morning" cannot be read backwards.
    assert "start_after means the class starts at or after" in prompt
    assert "end_before means it ends at or before" in prompt


@pytest.mark.parametrize("text,expected", [
    ("I have class MWF 10:00-10:50", True),
    ("busy Tuesdays and Thursdays", True),
    ("free after 3pm", True),
    ("I work mornings", True),
    ("what is topology about", False),
    ("tell me about COMP 140", False),
    ("show open seats in MATH 355", False),
    # Asking for a class at certain times is not a busy-time statement.
    ("give me an analyzing diversity class that meets in the mornings, "
     "monday wednesday and friday", False),
    ("Find me classes that meet MWF mornings", False),
    ("are there any courses on Tuesday afternoons?", False),
    # No request verb at all, as typed in the real chat.
    ("asian studies class in the morning of Monday Wednesday Friday", False),
    ("computer science course Tuesday Thursday afternoons", False),
    # A terse statement with exact times is still a busy time.
    ("class Monday 10 to 11am", True),
    ("class M 08:00-08:50", True),
    # ...unless the same message also says when the student is busy.
    ("I have class Monday 10:00 to 11:15. Find open CS courses", True),
    ("Find a class on Fridays, I also work Mondays 1-5pm", True),
])
def test_looks_like_schedule(text, expected):
    assert looks_like_schedule(text) is expected


def test_extractor_output_is_validated_and_deduplicated():
    reply = extraction([
        {"day": "w", "start": "10:00", "end": "11:15"},
        {"day": "M", "start": "10:00", "end": "11:15"},
        {"day": "M", "start": "10:00", "end": "11:15"},
    ])
    update = extract_busy_blocks(FakeLLM(extraction=[reply]), [], "busy MW")
    assert [(b.day, str(b.start)) for b in update.blocks] == \
        [("M", "10:00:00"), ("W", "10:00:00")]


def test_clearing_the_schedule(conn, session):
    session.busy_blocks = [BusyBlock("M", "10:00", "11:00")]
    llm = FakeLLM(extraction=[extraction([])], main=[say("Cleared.")])
    run_turn(session, "forget my schedule, start over", llm, conn)
    assert session.busy_blocks == []


# --------------------------------------------------------------------------
# Grounding: only what tools returned
# --------------------------------------------------------------------------

def test_ungrounded_crn_triggers_one_retry_then_passes(conn, session):
    llm = FakeLLM(main=[
        say("You should take [CRN 99999]."),
        say("I could not find a matching section."),
    ])
    events = []
    r = run_turn(session, "what should I take", llm, conn,
                 on_event=lambda k, p: events.append((k, p)))
    assert events == [("retry", ["CRN 99999"])]
    correction = llm.main_calls[1]["messages"][-1]
    assert correction["role"] == "user" and "CRN 99999" in correction["content"]
    assert r.reply == "I could not find a matching section."


def test_persistent_ungrounded_crn_is_removed_with_a_warning(conn, session):
    llm = FakeLLM(main=[say("Take [CRN 99999]."), say("Really, [CRN 99999].")])
    r = run_turn(session, "what should I take", llm, conn)
    assert "[CRN 99999]" not in r.reply
    assert "[unverified CRN removed]" in r.reply
    assert "Warning: I removed CRN 99999" in r.reply
    assert r.removed_references == ["CRN 99999"]


def test_bare_unknown_crn_is_also_caught(conn, session):
    llm = FakeLLM(main=[say("Take CRN 88888."), say("Take CRN 88888 now.")])
    r = run_turn(session, "what should I take", llm, conn)
    assert "[unverified CRN removed]" in r.reply


def test_invented_course_code_is_flagged(conn, session):
    llm = FakeLLM(main=[say("Take COMP 999."), say("COMP 999 is great.")])
    r = run_turn(session, "what should I take", llm, conn)
    assert "Warning: I could not verify COMP 999" in r.reply


def test_course_codes_from_tools_or_the_student_are_grounded(conn, session):
    llm = FakeLLM(main=[
        tool_call("get_course_sections", subject="COMP", number="140"),
        say("COMP 140 is offered; you also mentioned MATH 101."),
    ])
    r = run_turn(session, "is COMP 140 offered? I took MATH 101", llm, conn)
    assert "Warning" not in r.reply
    assert len(llm.main_calls) == 2          # no correction round


def test_find_ungrounded_ignores_non_subject_words(session):
    session.seen["22130"] = {"crn": "22130"}
    assert guard.find_ungrounded("Spring 2027 term, CRN 22130", session,
                                 subjects={"COMP"}) == []
    assert guard.find_ungrounded("take COMP 999", session,
                                 subjects={"COMP"}) == ["COMP 999"]
    assert guard.find_ungrounded("take ZZZZ 999", session,
                                 subjects={"COMP"}) == []


# --------------------------------------------------------------------------
# Citations and flags
# --------------------------------------------------------------------------

def _seen(session, crn, **kw):
    base = dict(term_code="202720", crn=crn, subject="COMP", number="140",
                section="002", title="COMPUTATIONAL THINKING",
                permission_required=False, restrictions=None,
                has_meetings=True,
                enrollment_as_of=datetime(2026, 11, 7, 15, 42))
    base.update(kw)
    record_seen(session, **base)


def test_marker_expands_to_subject_number_section_and_crn(session):
    _seen(session, "22130")
    out = guard.finalize("Take [CRN 22130] and [crn: 22130].", session)
    assert out.text.count("COMP 140 section 002 (CRN 22130)") == 2
    assert out.cited_crns == ["22130"]


def test_bare_crn_is_expanded_but_full_citation_is_left_alone(session):
    _seen(session, "22130")
    bare = guard.finalize("Take CRN 22130 today.", session)
    assert "COMP 140 section 002 (CRN 22130)" in bare.text
    full = guard.finalize("Take COMP 140 section 002 (CRN 22130).", session)
    assert full.text.count("COMP 140") == 1


def test_missing_flags_are_added_from_the_data(session):
    _seen(session, "30001", permission_required=True, has_meetings=False,
          restrictions="Must be enrolled in one of the following Level(s):; "
                       "Graduate")
    out = guard.finalize("Take [CRN 30001] for fun.", session)
    assert out.notes_added == 1
    notes = out.text.split("Notes:")[1]
    assert "permission is required to enroll" in notes
    assert "no scheduled meeting time" in notes
    assert "restrictions: Graduate students only" in notes


def test_flags_the_model_already_stated_are_not_repeated(session):
    _seen(session, "30001", permission_required=True, has_meetings=False,
          restrictions="Must be enrolled in one of the following Level(s):; "
                       "Graduate")
    text = ("- [CRN 30001] Seminar: permission required, graduate students "
            "only, and it has no scheduled meeting time.")
    out = guard.finalize(text, session)
    assert out.notes_added == 0 and "Notes:" not in out.text


def test_flags_are_checked_per_section(session):
    _seen(session, "30001", permission_required=True)
    _seen(session, "30002", permission_required=False)
    text = ("- [CRN 30002] Fine class.\n"
            "- [CRN 30001] Other class.\n"
            "  Needs nothing special.")
    out = guard.finalize(text, session)
    assert out.notes_added == 1
    assert "(CRN 30001): permission is required" in out.text.split("Notes:")[1]
    # Mentioned on a continuation line counts for its own bullet only.
    ok = guard.finalize("- [CRN 30001] Other class.\n  Permission required.",
                        session)
    assert ok.notes_added == 0


def test_real_section_without_meetings_and_with_permission_is_flagged(
        conn, session):
    rows = q(conn, """
        SELECT s.crn FROM sections s
        WHERE s.is_active AND s.group_id IS NULL AND s.permission_required
          AND NOT EXISTS (SELECT 1 FROM meetings m
                          WHERE m.term_code = s.term_code AND m.crn = s.crn)
        ORDER BY s.crn LIMIT 1""")
    if not rows:
        pytest.skip("no permission-only section without meetings")
    crn = rows[0]["crn"]
    llm = FakeLLM(main=[tool_call("get_section", crn=crn),
                        say(f"Look at [CRN {crn}].")])
    r = run_turn(session, f"tell me about {crn}", llm, conn)
    notes = r.reply.split("Notes:")[1].split("Data as of")[0]
    assert "permission is required" in notes
    assert "no scheduled meeting time" in notes
    envelope = r.tool_calls[0].result
    assert envelope["results"][0]["meetings"] == NO_MEETING
    assert any(NO_MEETING in n for n in envelope["notes"])


# --------------------------------------------------------------------------
# Tool results and tool errors
# --------------------------------------------------------------------------

def test_search_envelope_shape_and_restriction_summary(conn):
    env = dispatch(conn, Session(), "search_sections",
                   {"query": "calculus", "subject": "MATH", "limit": 5})
    assert env["term"] and env["term_label"] == "Spring 2027"
    assert env["data_as_of"].endswith(("CST", "CDT"))
    assert env["count"] == len(env["results"]) <= 5
    row = env["results"][0]
    assert row["cite_as"] == f"[CRN {row['crn']}]"
    for key in ("subject", "number", "section", "crn", "title", "instructors",
                "meetings", "open_seats", "permission_required",
                "restrictions"):
        assert key in row
    flat = json.dumps(env)
    assert "Must be enrolled in one of the following" not in flat
    assert any(r["restrictions"] in ("Graduate students only",
                                     "Undergraduate students only")
               for r in env["results"])
    json.dumps(env)


def test_sections_without_meetings_say_so_explicitly(conn):
    rows = q(conn, """
        SELECT s.crn FROM sections s WHERE s.is_active AND s.group_id IS NULL
          AND NOT EXISTS (SELECT 1 FROM meetings m WHERE m.term_code =
                          s.term_code AND m.crn = s.crn) LIMIT 1""")
    env = dispatch(conn, Session(), "get_section", {"crn": rows[0]["crn"]})
    assert env["results"][0]["meetings"] == NO_MEETING
    assert any(NO_MEETING in n for n in env["notes"])


def test_results_are_capped_and_marked_truncated(conn):
    env = dispatch(conn, Session(), "search_sections",
                   {"subject": "COMP", "limit": 100})
    assert env["count"] == config.MAX_RESULTS_CAP
    assert env["truncated"] is True
    default = dispatch(conn, Session(), "search_sections", {"subject": "COMP"})
    assert default["count"] == config.MAX_RESULTS


def test_string_arguments_from_small_models_are_coerced(conn):
    env = dispatch(conn, Session(), "search_sections", {
        "subject": "COMP", "level_min": "200", "level_max": "299.0",
        "open_seats_only": "true", "limit": "3"})
    assert 0 < env["count"] <= 3
    assert all(200 <= int(r["number"]) <= 299 and r["open_seats"] > 0
               for r in env["results"])
    as_json = dispatch(conn, Session(), "search_sections",
                       json.dumps({"subject": "MATH", "limit": 2}))
    assert as_json["count"] == 2


def test_search_tool_filters_by_wanted_days_and_times(conn):
    env = dispatch(conn, Session(), "search_sections", {
        "days": "MWF", "end_before": "12:00", "limit": 15})
    assert env["count"] > 0
    for r in env["results"]:
        assert r["meetings"] != NO_MEETING
        for part in r["meetings"].split("; "):
            letters, span = part.split()
            assert set(letters) <= set("MWF")
            assert span.split("-")[1] <= "12:00"
    assert any("meeting days/times" in n for n in env["notes"])
    assert "busy_blocks" not in json.dumps(tool_definitions())
    params = tool_definitions()[0]["function"]["parameters"]["properties"]
    assert {"days", "start_after", "end_before"} <= set(params)


def _meeting_spans(env):
    """``[(letters, start, end), ...]`` for every meeting in a search result."""
    spans = []
    for r in env["results"]:
        for part in r["meetings"].split("; "):
            letters, span = part.split()
            start, end = span.split("-")
            spans.append((letters, start, end))
    return spans


def test_time_of_day_morning_means_ends_by_noon(conn):
    env = dispatch(conn, Session(), "search_sections", {
        "days": "MWF", "time_of_day": "morning", "limit": 15})
    assert env["count"] > 0
    assert all(end <= "12:00" for _, _, end in _meeting_spans(env))
    assert env["meeting_filters"] == "days MWF; ends by 12:00"
    assert any("ends by 12:00" in n for n in env["notes"])


def test_time_of_day_afternoon_and_evening(conn):
    aft = dispatch(conn, Session(), "search_sections",
                   {"time_of_day": "Afternoon", "limit": 15})   # any case
    assert aft["count"] > 0
    assert all("12:00" <= s and e <= "17:00"
               for _, s, e in _meeting_spans(aft))
    eve = dispatch(conn, Session(), "search_sections",
                   {"time_of_day": "evening", "limit": 15})
    assert eve["count"] > 0
    assert all(s >= "17:00" for _, s, _ in _meeting_spans(eve))
    assert eve["meeting_filters"] == "starts at or after 17:00"


def test_explicit_time_overrides_that_side_of_time_of_day(conn):
    env = dispatch(conn, Session(), "search_sections", {
        "time_of_day": "morning", "end_before": "10:30", "limit": 15})
    assert env["meeting_filters"] == "ends by 10:30"
    assert all(end <= "10:30" for _, _, end in _meeting_spans(env))


def test_bad_time_of_day_is_a_tool_error_and_no_filter_means_no_echo(conn):
    err = dispatch(conn, Session(), "search_sections",
                   {"time_of_day": "midnight"})
    assert "time_of_day" in err["error"] and "morning" in err["error"]
    plain = dispatch(conn, Session(), "search_sections", {"subject": "COMP"})
    assert plain["meeting_filters"] is None


def test_time_of_day_is_in_the_tool_schema_with_fixed_values():
    props = tool_definitions()[0]["function"]["parameters"]["properties"]
    assert props["time_of_day"]["enum"] == ["morning", "afternoon", "evening"]


FAKE_SUBJECTS = [Subject("ASIA", "Asian Studies", 14),
                 Subject("COMP", "Computer Science", 214),
                 Subject("HIST", "History", 46),
                 Subject("MUSI", "Music", 30)]


@pytest.mark.parametrize("text,code", [
    ("Asian Studies", "ASIA"), ("asian studies", "ASIA"), ("ASIA", "ASIA"),
    ("asia", "ASIA"), ("Computer Science", "COMP"), ("comp", "COMP"),
    ("computer-science!", "COMP"), ("the history department", "HIST"),
    ("asian studies classes", "ASIA"), ("any music courses", "MUSI"),
])
def test_resolve_subject_text_matches_codes_and_names(text, code):
    assert resolve_subject_text(text, FAKE_SUBJECTS).code == code


@pytest.mark.parametrize("text", [
    "history of science",           # a topic that merely contains a name
    "computer science and poetry", "asian", "COMP 140", "poetry", "classes",
    "", "the",
])
def test_resolve_subject_text_ignores_everything_else(text):
    assert resolve_subject_text(text, FAKE_SUBJECTS) is None


def _subjects_in(env):
    return {r["subject"] for r in env["results"]}


def test_department_name_in_subject_argument_is_mapped_to_its_code(conn):
    env = dispatch(conn, Session(), "search_sections",
                   {"subject": "Asian Studies", "limit": 15})
    assert env["count"] > 0 and _subjects_in(env) == {"ASIA"}
    assert any("ASIA" in n and "Asian Studies" in n for n in env["notes"])
    # A plain code needs no note.
    plain = dispatch(conn, Session(), "search_sections",
                     {"subject": "asia", "limit": 15})
    assert _subjects_in(plain) == {"ASIA"}
    assert not any("Read subject" in n for n in plain["notes"])


def test_department_name_in_query_becomes_a_subject_filter(conn):
    """The 'asian studies' bug: the words are not in the course titles."""
    env = dispatch(conn, Session(), "search_sections",
                   {"query": "asian studies classes", "limit": 15})
    assert env["count"] > 0 and _subjects_in(env) == {"ASIA"}
    assert any("as the subject ASIA" in n for n in env["notes"])
    both = dispatch(conn, Session(), "search_sections",
                    {"query": "Asian Studies", "subject": "ASIA", "limit": 15})
    assert crns_of(both) == crns_of(env)    # the query was dropped, not ANDed


def test_topic_queries_are_left_alone(conn):
    env = dispatch(conn, Session(), "search_sections",
                   {"query": "poetry", "subject": "ENGL", "limit": 5})
    assert env["count"] > 0 and not any("subject" in n.lower()
                                        for n in env["notes"])


def test_several_departments_in_one_argument(conn):
    env = dispatch(conn, Session(), "search_sections",
                   {"subject": "COMP, math", "limit": 15,
                    "level_min": 200, "level_max": 399})
    assert _subjects_in(env) <= {"COMP", "MATH"}
    both = dispatch(conn, Session(), "search_sections", {
        "subject": "Computer Science and Mathematics", "limit": 15})
    assert _subjects_in(both) <= {"COMP", "MATH"}


def test_unknown_subject_is_an_error_that_lists_valid_ones(conn):
    err = dispatch(conn, Session(), "search_sections",
                   {"subject": "Underwater Basket Weaving"})
    assert "unknown subject" in err["error"]
    assert "ASIA (Asian Studies)" in err["error"]


def test_known_subject_with_no_sections_loaded_is_flagged(conn):
    from services import list_subjects
    empty = [s for s in list_subjects(conn) if s.active_sections == 0]
    if not empty:
        pytest.skip("every known subject has sections loaded")
    env = dispatch(conn, Session(), "search_sections",
                   {"subject": empty[0].code})
    assert env["count"] == 0
    assert any("No sections of" in n and "loaded" in n for n in env["notes"])


def test_prompt_and_schema_steer_departments_to_subject():
    prompt = " ".join(build_system_prompt("202720", []).split())
    assert "subject argument" in prompt and "not in query" in prompt
    props = tool_definitions()[0]["function"]["parameters"]["properties"]
    assert "department" in props["subject"]["description"]


def test_bad_wanted_time_is_a_tool_error(conn):
    err = dispatch(conn, Session(), "search_sections", {"days": "MXF"})
    assert "days must use" in err["error"]
    err = dispatch(conn, Session(), "search_sections", {"end_before": "noon"})
    assert "HH:MM" in err["error"]


def test_keyword_only_setting_never_touches_the_embedding_model(
        conn, monkeypatch):
    from services import embeddings

    def boom(*a, **k):
        raise AssertionError("the embedding model was used")
    monkeypatch.setattr(embeddings, "embed_query", boom)
    monkeypatch.setattr(embeddings, "embed_texts", boom)
    monkeypatch.setattr(config, "SEMANTIC", False)
    env = dispatch(conn, Session(), "search_sections",
                   {"query": "poetry", "limit": 3})
    assert "error" not in env


def test_search_does_not_wait_for_a_model_that_is_still_loading(
        conn, monkeypatch):
    from services import embeddings

    def boom(*a, **k):
        raise AssertionError("search waited for the embedding model")
    monkeypatch.setattr(embeddings, "embed_query", boom)
    monkeypatch.setattr(embeddings, "embed_texts", boom)
    monkeypatch.setattr(config, "SEMANTIC", True)
    monkeypatch.setattr(config, "SEMANTIC_LOADING", True)
    env = dispatch(conn, Session(), "search_sections",
                   {"query": "poetry", "limit": 3})
    assert "error" not in env and env["count"] > 0
    assert any("still loading" in n for n in env["notes"])
    # Without a query there is nothing to explain.
    plain = dispatch(conn, Session(), "search_sections", {"subject": "COMP"})
    assert not any("still loading" in n for n in plain["notes"])


def test_bad_tool_calls_return_errors_instead_of_raising(conn):
    s = Session()
    assert "level_min" in dispatch(conn, s, "search_sections",
                                   {"level_min": "abc"})["error"]
    assert "crn" in dispatch(conn, s, "get_section", {})["error"]
    assert "Unknown tool" in dispatch(conn, s, "book_flight", {})["error"]
    assert "valid JSON" in dispatch(conn, s, "get_section", "{oops")["error"]
    assert "object" in dispatch(conn, s, "get_section", [1])["error"]
    assert "non-empty list" in dispatch(
        conn, s, "seat_status", {"crns": []})["error"]
    assert "at most" in dispatch(conn, s, "seat_status", {
        "crns": [str(i) for i in range(50)]})["error"]


def test_model_sees_a_tool_error_and_can_retry(conn, session):
    llm = FakeLLM(main=[
        tool_call("search_sections", level_min="abc"),
        tool_call("search_sections", query="poetry", subject="ENGL"),
        say("Here you go."),
    ])
    r = run_turn(session, "poetry classes", llm, conn)
    assert "error" in r.tool_calls[0].result
    assert "error" not in r.tool_calls[1].result
    second_call_msgs = llm.main_calls[1]["messages"]
    assert "level_min" in second_call_msgs[-1]["content"]


def test_get_section_and_course_sections_and_seat_status(conn, session):
    env = dispatch(conn, session, "get_course_sections",
                   {"subject": "comp", "number": "140"})
    assert env["count"] > 0
    crn = env["results"][0]["crn"]
    assert all(r["subject"] == "COMP" and r["number"] == "140"
               for r in env["results"])
    detail = dispatch(conn, session, "get_section", {"crn": crn})
    row = detail["results"][0]
    assert row["description"] and "prerequisites" in row
    assert row["restrictions_raw"] is None or \
        row["restrictions_raw"].startswith("Must")
    seats = dispatch(conn, session, "seat_status", {"crns": [crn, "00000"]})
    assert [r["found"] for r in seats["results"]] == [True, False]
    assert seats["results"][0]["data_as_of"].endswith(("CST", "CDT"))
    assert any("00000" in n for n in seats["notes"])
    missing = dispatch(conn, session, "get_section", {"crn": "00000"})
    assert missing["results"] == [] and "No section" in missing["notes"][0]


def test_list_tools(conn, session):
    subjects = dispatch(conn, session, "list_subjects", {})
    assert any(r["code"] == "COMP" for r in subjects["results"])
    assert all(r["sections"] > 0 for r in subjects["results"])
    groups = dispatch(conn, session, "list_distribution_groups", {})
    assert groups["results"] and groups["results"][0]["name"].startswith(
        "Distribution Group")


def test_conflict_tool_uses_the_database_not_the_model(conn, session):
    pair = q(conn, """
        SELECT a.crn AS a, b.crn AS b
        FROM meetings a JOIN meetings b
          ON a.term_code = b.term_code AND a.day_of_week = b.day_of_week
         AND a.crn < b.crn
        WHERE a.start_time < b.end_time AND b.start_time < a.end_time
          AND a.start_date <= b.end_date AND b.start_date <= a.end_date
        LIMIT 1""")
    if not pair:
        pytest.skip("no overlapping sections in the data")
    p = pair[0]
    env = dispatch(conn, session, "check_schedule_conflicts",
                   {"crns": [p["a"], p["b"]]})
    assert env["no_conflicts"] is False
    assert {tuple(r["crns"]) for r in env["results"]} >= {(p["a"], p["b"])}
    assert {p["a"], p["b"]} <= set(session.seen)       # recorded as grounded
    one = dispatch(conn, session, "check_schedule_conflicts",
                   {"crns": [p["a"]]})
    assert one["no_conflicts"] is True
    unknown = dispatch(conn, session, "check_schedule_conflicts",
                       {"crns": [p["a"], "00000"]})
    assert unknown["unknown_crns"] == ["00000"]


def test_conflict_tool_notes_sections_without_meetings(conn, session):
    rows = q(conn, """
        SELECT s.crn FROM sections s WHERE s.is_active AND s.group_id IS NULL
          AND NOT EXISTS (SELECT 1 FROM meetings m WHERE m.term_code =
                          s.term_code AND m.crn = s.crn) LIMIT 1""")
    env = dispatch(conn, session, "check_schedule_conflicts",
                   {"crns": [rows[0]["crn"]]})
    assert env["no_meeting_time_crns"] == [rows[0]["crn"]]
    assert any(NO_MEETING in n for n in env["notes"])


# --------------------------------------------------------------------------
# Loop limits, history and session state
# --------------------------------------------------------------------------

def test_step_limit_stops_runaway_tool_loops(conn, session):
    call = tool_call("list_distribution_groups")
    llm = FakeLLM(main=[call, call, call, say("Here is what I found.")])
    r = run_turn(session, "groups?", llm, conn, max_steps=3)
    assert r.hit_step_limit and r.steps == 3
    assert len(r.tool_calls) == 3
    assert r.reply.startswith("Here is what I found.")
    last = llm.main_calls[-1]
    assert last["tools"] is None                  # forced to answer
    assert "Answer now" in last["messages"][-1]["content"]


def test_history_keeps_answer_without_footer_and_trims(conn, session):
    llm = FakeLLM(main=[
        tool_call("search_sections", query="poetry", subject="ENGL"),
        lambda m: say(f"Consider [CRN {first_crn(m)}]."),
    ])
    r = run_turn(session, "poetry classes", llm, conn)
    assert "Data as of" in r.reply
    stored = session.messages[-1]
    assert stored["role"] == "assistant" and "Data as of" not in stored["content"]

    # Old tool output is shortened and the history stays bounded.
    from agent.core import HISTORY_LIMIT, OLD_TOOL_RESULT_CHARS
    session.messages += [{"role": "user", "content": f"q{i}"} if i % 2 == 0
                         else {"role": "assistant", "content": f"a{i}"}
                         for i in range(HISTORY_LIMIT * 2)]
    llm2 = FakeLLM(main=[say("Hi.")])
    run_turn(session, "hello again", llm2, conn)
    sent = llm2.main_calls[0]["messages"]
    assert sent[0]["role"] == "system" and sent[1]["role"] == "user"
    assert len(sent) <= HISTORY_LIMIT + 2
    for m in session.messages:
        if m["role"] == "tool":
            assert len(m["content"]) <= OLD_TOOL_RESULT_CHARS + 20


def test_session_round_trips_through_json(conn, session, comp_meeting):
    block = block_dict(comp_meeting)
    llm = FakeLLM(extraction=[extraction([block])],
                  main=[tool_call("search_sections", subject="COMP", limit=3),
                        lambda m: say(f"Try [CRN {first_crn(m)}].")])
    run_turn(session, f"class {block['day']} {block['start']}-{block['end']}",
             llm, conn)
    again = Session.from_dict(json.loads(json.dumps(session.to_dict())))
    assert again.busy_blocks == session.busy_blocks
    assert again.seen == session.seen and again.messages == session.messages
    assert again.known_courses == session.known_courses
    assert again.term_code == session.term_code


# --------------------------------------------------------------------------
# Ollama client (no server needed: httpx mock transport)
# --------------------------------------------------------------------------

def client_with(handler, **kw):
    c = OllamaClient(model="qwen3:8b", host="http://ollama.test", **kw)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def test_chat_sends_tools_format_options_and_returns_the_message():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"message": {
            "role": "assistant", "content": "hi"}})

    c = client_with(handler, num_ctx=4096)
    msg = c.chat([{"role": "user", "content": "x"}], tools=[{"t": 1}],
                 format={"type": "object"}, think=False)
    assert msg == {"role": "assistant", "content": "hi"}
    body = seen["body"]
    assert seen["url"] == "http://ollama.test/api/chat"
    assert body["model"] == "qwen3:8b" and body["stream"] is False
    assert body["tools"] == [{"t": 1}] and body["format"] == {"type": "object"}
    assert body["think"] is False
    assert body["options"] == {"temperature": 0, "num_ctx": 4096}


def test_chat_omits_optional_fields_and_reports_errors():
    bodies = []

    def ok(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "x"}})

    client_with(ok).chat([{"role": "user", "content": "x"}])
    assert not {"tools", "format", "think"} & set(bodies[0])

    with pytest.raises(LLMError, match="500"):
        client_with(lambda r: httpx.Response(500, text="boom")).chat([])
    with pytest.raises(LLMError, match="no message"):
        client_with(lambda r: httpx.Response(200, json={})).chat([])

    def refuse(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(LLMError, match="request failed"):
        client_with(refuse).chat([])


def test_check_explains_how_to_fix_a_missing_server_or_model():
    def refuse(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(LLMError, match="ollama serve"):
        client_with(refuse).check()

    def tags(names):
        return lambda r: httpx.Response(200, json={
            "models": [{"name": n} for n in names]})

    with pytest.raises(LLMError, match="ollama pull qwen3:8b"):
        client_with(tags(["llama3.1:8b"])).check()
    client_with(tags(["qwen3:8b"])).check()          # ready: no error

    c = OllamaClient(model="llama3.1", host="http://ollama.test")
    c._http = httpx.Client(transport=httpx.MockTransport(tags(["llama3.1:latest"])))
    c.check()                                         # ":latest" is implied


# --------------------------------------------------------------------------
# Optional: a real local model
# --------------------------------------------------------------------------

def _ollama_ready():
    try:
        OllamaClient().check()
        return True
    except LLMError:
        return False


@pytest.mark.ollama
@pytest.mark.skipif(not _ollama_ready(),
                    reason="Ollama is not running with the default model")
def test_live_model_end_to_end(conn):
    """Asserts only what code guarantees, not the model's wording."""
    llm = OllamaClient()
    session = Session()
    prompts = [
        "I have class Monday and Wednesday 10:00am to 11:15am. Find open "
        "computer science courses at the 200 level.",
        "Anything about machine learning that is not graduate only?",
        "Tell me about COMP 140.",
    ]
    for text in prompts:
        r = run_turn(session, text, llm, conn)
        assert r.reply
        if r.tool_calls:
            assert "Data as of" in r.reply and "ESTHER" in r.reply
        assert guard.find_ungrounded(r.reply, session) == []
        assert "[CRN" not in r.reply
    assert {(b.day, str(b.start)) for b in session.busy_blocks} >= {
        ("M", "10:00:00"), ("W", "10:00:00")}
