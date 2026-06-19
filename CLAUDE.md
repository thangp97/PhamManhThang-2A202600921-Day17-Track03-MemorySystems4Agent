# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A teaching lab (Stage 2, Track 3, Day 17: "Memory Systems for AI Agent"). The goal is to build and benchmark two agents to study the trade-off between long-term recall, response quality, token cost, and memory-system complexity:

- **Baseline Agent** (`agent_baseline.py`) — short-term memory within a single thread only. Deliberately must NOT recall facts across new threads/sessions.
- **Advanced Agent** (`agent_advanced.py`) — three memory layers: short-term (within-thread), persistent (`User.md` per user), and compact memory (summarizes old messages once a long thread crosses a token threshold).

The `src/` tree is a **student scaffold**: every function/method currently raises `NotImplementedError` or is a TODO stub. Implementing them is the assignment. `Guide.md` (step-by-step) and `Rubric.md` (grading) at the repo root define the intended behavior and are the source of truth for requirements.

## Commands

Run everything from inside `src/` — modules import each other by top-level name (`from config import ...`, `from memory_store import ...`), so the working directory must be `src/` or it must be on `PYTHONPATH`. There is no package `__init__.py`.

```bash
# Environment (Python >= 3.11)
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install langchain langgraph langchain-openai langchain-google-genai \
  langchain-anthropic langchain-ollama langchain-openrouter python-dotenv tabulate pytest

# Run the benchmark (from src/)
python benchmark.py

# Tests
pytest                       # run from src/
pytest test_agents.py::test_compact_trigger    # single test
```

There is no linter or build step configured.

## Architecture

Module layering (each imports only from the ones above it):

1. `model_provider.py` — `ProviderConfig` dataclass + `build_chat_model()`. Must support `openai`, `custom` (OpenAI-compatible base URL), `gemini`, `anthropic`, `ollama`, `openrouter`. `normalize_provider()` maps aliases (e.g. `anthorpic` → `anthropic`).
2. `config.py` — `LabConfig` dataclass + `load_config()`. Holds repo/`data`/`state` paths, compact-memory knobs (`compact_threshold_tokens`, `compact_keep_messages`), and provider config for the main model and the judge model. Reads provider/keys from env (and optionally `.env`).
3. `memory_store.py` — the core of the lab: `estimate_tokens()` (heuristic, ~chars/4 is fine), `UserProfileStore` (maps a user id to one `User.md` file with read/write/edit/size), `extract_profile_updates()` (regex-extract stable facts: name, location, profession, response style, preferences — and skip question-only turns), and `CompactMemoryManager` (keeps recent messages, summarizes older ones past the threshold, counts compactions).
4. `agent_baseline.py` / `agent_advanced.py` — the two agents (see above).
5. `benchmark.py` — runs both agents over both datasets and prints a comparison table.
6. `test_agents.py` — pytest coverage for `User.md` I/O, compaction trigger, cross-session recall, and prompt-load reduction on long threads.

### Offline-first design (important)

Each agent has BOTH a deterministic offline path (`_reply_offline` / `_offline_response`) and an optional live LangChain/LangGraph path (`_maybe_build_langchain_agent`). The offline path is what makes benchmarks and tests reproducible without API keys; `force_offline=True` forces it. When implementing, keep the offline path fully functional on its own — live agents are an enhancement gated by dependency/key availability.

### Token accounting model

The benchmark distinguishes two token metrics that drive the lab's central conclusion — track them separately:

- **Agent tokens only** — tokens the agent generates in its replies.
- **Prompt tokens processed** — context dragged into each turn (`User.md` + compact summary + recent messages for advanced; growing raw history for baseline).

Expected story the implementation should demonstrate: on short threads Advanced can cost *more* than Baseline; on very long threads compact memory mainly reduces *prompt tokens processed*, giving Advanced the advantage. Don't "fix" code that reproduces this — it's the intended result.

## Data

Datasets live at the repo **root** under `data/` (not in `src/`):

- `data/conversations.json` — standard benchmark input.
- `data/advanced_long_context.json` — long-context stress benchmark (must be long enough to expose Baseline's growing prompt cost).

Each conversation is `{ "id", "user_id", "turns": [...] }` with Vietnamese turns. `load_config()` resolves the data dir relative to the repo root (parent of `src/`), so pass `Path(__file__).resolve().parent.parent` as the base dir as `benchmark.py` already does.

## Conventions

- Persistent state (`User.md` profiles, etc.) is written under `state/`, which is gitignored — never commit it. `.env` is also gitignored.
- Code and benchmark output are English-facing; the benchmark **data and agent responses are Vietnamese** (the recall questions and expected facts are in Vietnamese).
- Keep naming consistent between Baseline and Advanced (same method names: `reply`, `token_usage`, `prompt_token_usage`, `compaction_count`) — the benchmark drives both through the same interface.
