"""Settings for the assistant, read from environment variables.

``OLLAMA_HOST`` and ``AGENT_MODEL`` are the two you are likely to change.
Everything is read once at import time; the CLI also lets ``--model``
override the model.
"""

import os

# Where the local Ollama server listens.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# A recent instruct model that supports tool calling. 8B fits an Apple M1
# with 16 GB; alternatives to compare: llama3.1:8b, qwen3:14b (tight).
MODEL = os.environ.get("AGENT_MODEL", "qwen3:8b")

# Context window requested from Ollama. Tool results are kept compact so an
# 8k window is enough for a normal conversation.
NUM_CTX = int(os.environ.get("AGENT_NUM_CTX", "8192"))

# Seconds to wait for one model reply (first call loads the model).
TIMEOUT = float(os.environ.get("AGENT_TIMEOUT", "300"))

# Most model/tool round trips in one user turn before giving up.
MAX_STEPS = 6

# Sections returned to the model per search by default, and the hard cap.
MAX_RESULTS = 8
MAX_RESULTS_CAP = 15

# Semantic (meaning-based) search needs the local embedding model, which is
# slow to load once per run. Set AGENT_SEMANTIC=0 (or pass --no-semantic) to
# use keyword search only. The CLI may switch this off if the model fails.
SEMANTIC = os.environ.get("AGENT_SEMANTIC", "1").strip().lower() \
    not in ("0", "false", "no", "off")

# True while the CLI is still loading the embedding model in the background.
# Searches made in that window use keyword matching instead of waiting.
SEMANTIC_LOADING = False

# Rice reports enrollment times in local (Central) time.
TIMEZONE = "America/Chicago"
