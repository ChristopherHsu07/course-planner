"""Terminal chat with the course assistant: ``python -m agent``.

Needs the Docker database (``DATABASE_URL`` set) and Ollama running with the
model pulled. Commands inside the chat: ``/blocks`` shows the busy times it
has on file, ``/reset`` starts over, ``/quit`` exits. ``--debug`` prints every
tool call and its result.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import threading

from db.load import connect

from . import config
from .core import ToolCall, run_turn
from .llm import LLMError, OllamaClient
from .schedule import describe_blocks
from .session import Session

HELP = "Commands: /blocks  /reset  /help  /quit"


def _print_event(kind: str, payload) -> None:
    """Debug output for ``run_turn`` events."""
    if kind == "blocks":
        print(f"  [busy times] {payload}")
    elif kind == "retry":
        print(f"  [guard] unsupported references {payload}; asking the "
              "model to fix them")
    elif kind == "tool":
        call: ToolCall = payload
        print(f"  [tool] {call.name}({json.dumps(call.arguments)})")
        shown = json.dumps(call.result, indent=2, default=str)
        print("\n".join("    " + line for line in shown.splitlines()[:60]))
        if len(shown.splitlines()) > 60:
            print("    ...")


def _warm_up_in_background() -> None:
    """Load the embedding model while the student types the first question.

    Importing it can take minutes on a slow machine. Until it is ready,
    searches use keyword matching (the tool tells the model so) instead of
    making the student wait. If the model cannot load, keyword search is
    used for the rest of the run.
    """
    def work() -> None:
        try:
            from services import embeddings
            embeddings.warm_up()
            print("\n[search] meaning-based search is ready.", flush=True)
        except Exception as e:                      # missing package, etc.
            config.SEMANTIC = False
            print(f"\n[search] semantic search is unavailable "
                  f"({e.__class__.__name__}); using keyword search.",
                  flush=True)
        finally:
            config.SEMANTIC_LOADING = False

    print("Loading the search model in the background; until it is ready, "
          "searches use keywords only (--no-semantic skips it).")
    config.SEMANTIC_LOADING = True
    threading.Thread(target=work, daemon=True).start()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=config.MODEL,
                    help=f"Ollama model (default {config.MODEL})")
    ap.add_argument("--host", default=config.OLLAMA_HOST,
                    help="Ollama server URL")
    ap.add_argument("--term", default=None,
                    help="term code (default: the latest loaded term)")
    ap.add_argument("--debug", action="store_true",
                    help="print each tool call and its result")
    ap.add_argument("--no-semantic", action="store_true",
                    help="keyword search only (skips loading the embedding "
                         "model, which is slow the first time)")
    args = ap.parse_args(argv)
    if args.no_semantic:
        config.SEMANTIC = False

    if not os.environ.get("DATABASE_URL"):
        print("DATABASE_URL is not set. Example:\n  export DATABASE_URL="
              "postgresql://planner:planner@localhost:5433/rice_planner")
        return 1
    llm = OllamaClient(model=args.model, host=args.host)
    try:
        llm.check()
    except LLMError as e:
        print(e)
        return 1

    session = Session(term_code=args.term)
    hook = _print_event if args.debug else None
    print(f"Course assistant ({args.model}). {HELP}")
    if config.SEMANTIC:
        _warm_up_in_background()
    with connect() as conn:
        while True:
            try:
                text = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not text:
                continue
            if text in ("/quit", "/exit"):
                return 0
            if text == "/help":
                print(HELP)
                continue
            if text == "/reset":
                session = Session(term_code=args.term)
                print("Starting over.")
                continue
            if text == "/blocks":
                print("Busy times on file:", describe_blocks(session.busy_blocks))
                continue
            before = copy.deepcopy(session.to_dict())
            if not args.debug:
                print("  (thinking...)", end="", flush=True)
            try:
                result = run_turn(session, text, llm, conn, on_event=hook)
            except LLMError as e:
                session = Session.from_dict(before)     # drop the failed turn
                print(f"\n[model error] {e}")
                continue
            finally:
                conn.rollback()                         # end the read transaction
                if not args.debug:
                    print("\r" + " " * 16 + "\r", end="", flush=True)
            print(f"\nassistant> {result.reply}")
            if args.debug:
                print(f"  [turn] {result.steps} model call(s), "
                      f"{len(result.tool_calls)} tool call(s)"
                      + (", step limit reached" if result.hit_step_limit
                         else ""))


if __name__ == "__main__":
    sys.exit(main())
