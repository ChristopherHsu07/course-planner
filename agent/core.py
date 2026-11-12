"""One conversational turn: extract busy times, run tools, check the answer.

``run_turn`` is a function of its arguments only (the session, the user's
text, a model, a database connection): no printing, no globals. A terminal
program or a web endpoint can call it and show ``TurnResult.reply``.

Turn flow::

    user text -> schedule extraction (structured JSON, validated)
              -> [question?] return it, run nothing else
              -> model <-> tools loop (at most ``MAX_STEPS`` round trips)
              -> guard: verify references (one retry), expand citations,
                 add flags and the freshness footer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from services import get_data_freshness, list_subjects
from services.db import resolve_term

from . import config, guard
from .llm import LLM
from .prompts import build_system_prompt
from .schedule import describe_blocks, extract_busy_blocks
from .session import Session
from .tools import dispatch, tool_definitions

# Messages of history kept for the model (the system message is extra).
HISTORY_LIMIT = 30
# Characters kept of a tool result from an earlier turn.
OLD_TOOL_RESULT_CHARS = 300

FALLBACK_REPLY = ("I could not put together an answer. Please try rephrasing "
                  "your question.")

EventHook = Callable[[str, Any], None]


@dataclass
class ToolCall:
    """One tool the model called this turn, with its result."""

    name: str
    arguments: Any
    result: dict


@dataclass
class TurnResult:
    """What a turn produced."""

    reply: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    asked_clarification: bool = False
    steps: int = 0
    hit_step_limit: bool = False
    cited_crns: list[str] = field(default_factory=list)
    removed_references: list[str] = field(default_factory=list)


def _set_system(session: Session) -> None:
    system = {"role": "system", "content": build_system_prompt(
        session.term_code, session.busy_blocks)}
    if session.messages and session.messages[0].get("role") == "system":
        session.messages[0] = system
    else:
        session.messages.insert(0, system)


def _compact_history(session: Session) -> None:
    """Keep the conversation inside the model's context window.

    Tool results from earlier turns are cut short (the assistant's answer
    already summarised them) and only the most recent messages are kept,
    never starting on an orphaned tool message.
    """
    msgs = session.messages
    for m in msgs[1:]:
        if m.get("role") == "tool" and len(m.get("content", "")) > \
                OLD_TOOL_RESULT_CHARS:
            m["content"] = m["content"][:OLD_TOOL_RESULT_CHARS] + " ...[cut]"
    body = msgs[1:]
    if len(body) > HISTORY_LIMIT:
        body = body[-HISTORY_LIMIT:]
        while body and body[0].get("role") != "user":
            body = body[1:]
        session.messages = [msgs[0], *body]


def _clean(message: dict) -> dict:
    """The parts of a model message worth keeping in history."""
    out = {"role": "assistant", "content": message.get("content") or ""}
    if message.get("tool_calls"):
        out["tool_calls"] = message["tool_calls"]
    return out


def _tool_call_parts(call: dict) -> tuple[str, Any]:
    fn = call.get("function") or {}
    return fn.get("name") or "", fn.get("arguments") or {}


def run_turn(session: Session, user_text: str, llm: LLM, conn, *,
             max_steps: int = config.MAX_STEPS,
             on_event: Optional[EventHook] = None) -> TurnResult:
    """Handle one user message and return the reply.

    ``on_event(kind, payload)`` is called for ``"blocks"`` (busy blocks after
    extraction), ``"tool"`` (a ``ToolCall``) and ``"retry"`` (a grounding
    correction); it is for debugging output and may be ``None``. Raises
    ``LLMError`` if the model server fails.
    """
    emit = on_event or (lambda kind, payload: None)
    session.tools_called_this_turn = 0
    session.note_text(user_text)
    if session.term_code is None:
        session.term_code = resolve_term(conn, None)

    # 1. Schedule first: validated blocks, or a question, before any tools.
    update = extract_busy_blocks(llm, session.busy_blocks, user_text)
    _set_system(session)
    if update.clarification:
        session.messages += [{"role": "user", "content": user_text},
                             {"role": "assistant",
                              "content": update.clarification}]
        return TurnResult(reply=update.clarification, asked_clarification=True)
    if update.ran:
        session.busy_blocks = update.blocks
        _set_system(session)
        emit("blocks", describe_blocks(session.busy_blocks))

    _compact_history(session)
    session.messages.append({"role": "user", "content": user_text})

    subjects = {s.code for s in list_subjects(conn)}
    calls: list[ToolCall] = []
    corrections_left = 1
    final_text: Optional[str] = None
    bad: list[str] = []
    steps = 0
    hit_limit = False

    # 2. The model <-> tools loop.
    while True:
        if steps >= max_steps:
            hit_limit = True
            session.messages.append({"role": "user", "content":
                "Answer now using only the tool results above."})
            reply = llm.chat(session.messages, think=False)
            session.messages.append(_clean(reply))
            final_text = reply.get("content") or ""
            bad = guard.find_ungrounded(final_text, session, subjects)
            break
        steps += 1
        reply = llm.chat(session.messages, tools=tool_definitions(), think=False)
        session.messages.append(_clean(reply))
        tool_calls = reply.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                name, arguments = _tool_call_parts(tc)
                result = dispatch(conn, session, name, arguments)
                call = ToolCall(name, arguments, result)
                calls.append(call)
                session.tools_called_this_turn += 1
                emit("tool", call)
                session.messages.append({
                    "role": "tool", "tool_name": name,
                    "content": json.dumps(result, default=str)})
            continue

        text = reply.get("content") or ""
        bad = guard.find_ungrounded(text, session, subjects)
        if bad and corrections_left:
            corrections_left -= 1
            emit("retry", bad)
            session.messages.append({"role": "user",
                                     "content": guard.correction_message(bad)})
            continue
        final_text = text
        break

    # 3. Guard: fix what is still wrong, add flags and the footer.
    final_text = (final_text or "").strip() or FALLBACK_REPLY
    unverified_courses = [b for b in bad if not b.startswith("CRN")]
    final_text = guard.warn_unverified_courses(final_text, unverified_courses)
    freshness = get_data_freshness(conn, session.term_code)
    result = guard.finalize(final_text, session, freshness,
                            used_tools=bool(calls))

    # History keeps the answer without the footer, so the model does not
    # learn to write one itself.
    session.messages[-1] = {"role": "assistant", "content": result.body}
    return TurnResult(reply=result.text, tool_calls=calls, steps=steps,
                      hit_step_limit=hit_limit, cited_crns=result.cited_crns,
                      removed_references=result.removed)
