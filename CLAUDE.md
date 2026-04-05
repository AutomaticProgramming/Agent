# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based LLM Agent with short-term and long-term memory systems, built on the OpenAI-compatible API. The project is educational — a learning-by-doing exploration of Agent architecture, iterated across three versions.

## Code Architecture

```
agent_1.py  → Basic Agent (bash/read/write tools, no memory)
agent_2.py  → + Short-term memory (auto-summarization) + Long-term memory (persistent JSON)
agent_3.py  → + Multi-turn conversation (AgentSession class, REPL loop)
```

### Key Components

- **`MemoryManager`** — manages long-term memory persistence (`./memory/long_term.json`) and short-term memory summarization.
  - `save(key, value)` — upsert key-value entries
  - `search(query)` — fuzzy-match against keys
  - `summarize(messages, client)` — LLM-based compression of conversation history

- **`AgentSession` (agent_3)** — cross-turn session manager. `step(user_message)` appends user input, runs the tool-calling loop, returns final answer. Preserves full message history across REPL turns.

- **Tool loop** — each LLM call includes a `tools` list; the agent iteratively resolves `tool_calls` until the model returns a final answer or `max_iterations` is reached.

- **Short-term memory** — `messages` list capped at `SHORT_MEMORY_LIMIT` (15); exceeded triggers LLM summarization to compress history.

- **Long-term memory** — JSON file at `./memory/long_term.json`, injected into system prompt at session start.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key (required) |
| `OPENAI_BASE_URL` | Custom API endpoint (optional, for local/compatible LLMs) |
| `OPENAI_MODEL` | Model name (default in agent_3: `DeepSeek-V3.2`) |

### How to Run

```bash
# Single task
python agent_3.py "你的任务描述"

# Interactive REPL mode
python agent_3.py
```

The `util/` and `png/` directories contain auxiliary files. Markdown docs (`从零开始理解Agent(一/二/三).md`) contain Chinese-language blog posts documenting the learning journey.
