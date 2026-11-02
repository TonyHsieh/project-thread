# Project Thread Application Layer (`app/`)

This directory contains the core application modules for `project-thread-api/v1`.

## Modules Overview
- **[`domain/`](domain/)**: Immutable dataclasses and domain exceptions (`Message`, `GenerationResult`, `TextStream`, `SearchResult`, `IndexManifest`, `ToolDefinition`, `ToolCall`, `ToolResult`, `ModelError`).
- **[`config.py`](config.py)**: Provider discovery (`azure` / `openai`), credentials isolation, and model resolution.
- **[`textgen.py`](textgen.py)**: Stateless text generation boundary (`generate_text`, `generate_text_stream`).
- **[`chat.py`](chat.py)**: Multi-turn conversational adapter (`respond`, `respond_stream`, `chat_loop`).
- **[`search.py`](search.py)**: Semantic corpus search engine (`search_corpus`).
- **[`tools.py`](tools.py)**: Tool registry, schema validator, and safe executor (`call_tool`, `get_openai_tools`).
- **[`plan.md`](plan.md)**: Persistent project roadmap, completed checklists, and design decision logs.

For complete documentation on each Python function and contract invariant, see [`../docs/`](../docs/) and the root [`../README.md`](../README.md).

