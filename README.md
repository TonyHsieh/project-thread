# Project Thread (`project-thread-api/v1`)

A robust, modular, and contract-governed AI application suite built for the AI Engineering Path (v0 in Phase 1 through capstone).

Centered around clean boundaries, explicit state management, and strict separation of concerns:
- **Stateless Generation:** [`app/textgen.py`](app/textgen.py)
- **Conversational Adapter:** [`app/chat.py`](app/chat.py)
- **Knowledge Retrieval:** [`app/search.py`](app/search.py)
- **Tool Execution & Registry:** [`app/tools.py`](app/tools.py)
- **Contract Models:** [`app/domain/models.py`](app/domain/models.py)

---

## Architecture Overview

```mermaid
flowchart TD
    User([User / Caller]) --> Chat["app/chat.py\n(respond, respond_stream, chat_loop)"]
    Chat -->|1. Optional Search| Search["app/search.py\n(search_corpus)"]
    Search -->|2. SearchResult[]| Chat
    Chat -->|3. context=[r.text]| TextGen["app/textgen.py\n(generate_text, generate_text_stream)"]
    TextGen -->|4. Calls API| Provider["OpenAI / Azure OpenAI Backend"]
    Provider -->|5. Chunks / Response| TextGen
    TextGen -->|6. GenerationResult / TextStream| Chat
    Chat -->|7. Model emits ToolCall| Tools["app/tools.py\n(call_tool)"]
    Tools -->|8. Validates Schema & Executes| Registry["ToolRegistry\n(search_corpus, calculate, get_current_time)"]
    Registry -->|9. ToolResult(ok=True/False)| Chat
```

---

## Detailed Function & Component Documentation

Detailed API contracts, parameters, invariants, and examples for every core function are documented in [`docs/`](docs/):

| Component | Target File | Documentation |
| :--- | :--- | :--- |
| **Generation** | `generate_text(...)` | [`docs/generate_text.md`](docs/generate_text.md) |
| **Streaming** | `generate_text_stream(...)` | [`docs/generate_text_stream.md`](docs/generate_text_stream.md) |
| **Chat Response** | `respond(...)` | [`docs/respond.md`](docs/respond.md) |
| **Chat Streaming** | `respond_stream(...)` | [`docs/respond_stream.md`](docs/respond_stream.md) |
| **Interactive Console** | `chat_loop(...)` | [`docs/chat_loop.md`](docs/chat_loop.md) |
| **Corpus Retrieval** | `search_corpus(...)` | [`docs/search_corpus.md`](docs/search_corpus.md) |
| **Tool Execution** | `call_tool(...)` | [`docs/call_tool.md`](docs/call_tool.md) |
| **Tool Registry** | `register_tool(...)` | [`docs/register_tool.md`](docs/register_tool.md) |
| **OpenAI Tool Export** | `get_openai_tools(...)` | [`docs/get_openai_tools.md`](docs/get_openai_tools.md) |
| **Domain Contracts** | Models, Invariants & v3.8 Migration | [`docs/domain_models.md`](docs/domain_models.md) |

---

## Core Invariants & Architecture Rules

1. **One-Module-One-Job Seam:**
   - `textgen.py` is the only module that interfaces with model backends. It never opens corpus files, manages session history, or queries vector indices.
   - Retrieval (`search.py`) is a caller-side step; generation only receives strings via `context`.
2. **Migration (v3.8) — `call_uid` Correlation:**
   - Unified single correlation key across `ToolCall.call_uid`, `ToolResult.call_uid`, and `Message.call_uid`.
   - Adheres to the *"one value, one name, three fields"* rule.
   - `call_uid` is preserved unchanged across success and failure paths (`ok = False`).
3. **The Model Proposes; Code Disposes:**
   - Model-generated tool calls are untrusted input. `call_tool` strictly validates payloads against declared JSON Schemas, rejects undeclared properties, and captures runtime errors as `ToolResult.ok = False` (never crashing the process).
4. **Context Isolation (Approach B):**
   - Retrieved passages are framed under `<context>` XML tags with an anti-instruction injection guard, preventing corpus text from masquerading as system instructions.
5. **The Null Rule:**
   - Unknown token counts and prices are `None`, never `0` or estimated.
6. **Stream Lifecycle (Option A):**
   - Accessing `stream.result` before exhausting all chunks from `generate_text_stream` or `respond_stream` raises a `RuntimeError`.

---

## Quickstart & CLI Usage

### 1. Environment Setup
Configure your API credentials in `.env`:
```bash
# OpenAI Provider
OPENAI_API_KEY=your_key_here
OPENAI_MODEL_ID=gpt-4o

# Or Azure OpenAI Provider
AI_PROVIDER=azure
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
```

### 2. Running Interactive Chat
```bash
# Pure multi-turn conversational chat
python app/chat.py

# Chat with live grounded corpus retrieval (RAG preview)
python app/chat.py --search
```
*In-session commands:*
- `/reset` — Clears memory.
- `/search` — Toggles active corpus retrieval on and off.
- `/exit` — Exits session.

### 3. Running Semantic Corpus Search
```bash
# Default search query
python app/search.py

# Custom search query
python app/search.py "transformers and self-attention"
```
*Supports recursive ingestion of `.txt`, `.md`, `.json`, and `.pdf` with incremental `.corpus_cache.json` caching.*

### 4. Running the Tool CLI
```bash
# List all registered tools and their enabled status
python app/tools.py list

# Run standalone self-test
python app/tools.py test

# Execute a tool directly from CLI
python app/tools.py call calculate "expression=25*4+17"
python app/tools.py call get_current_time "{}"
python app/tools.py call search_corpus "query=tokens top_k=2"

# Enable / Disable tools dynamically in tools_config.json
python app/tools.py disable calculate
python app/tools.py enable calculate
```

---

## Testing & Quality Assurance

Comprehensive unit, boundary, and invariant test suite using Python's standard `unittest` framework:
```bash
python -m unittest discover tests
```
*51 tests passing across `tests/unit/test_textgen.py`, `tests/unit/test_chat.py`, `tests/unit/test_search.py`, and `tests/unit/test_tools.py`.*

---

## Roadmap

Tracked in [`app/plan.md`](app/plan.md):
- [x] Phase 1–5: `textgen.py` & Invariant Models
- [x] Phase 6–8: `chat.py` Multi-turn Loop & `/reset`
- [x] Phase 9: `search.py` Whole-Document Semantic Retrieval
- [x] Phase 11: Grounded Generation Handoff
- [x] Phase 12: Tool API & Function Calling (`call_tool`) + Migration (v3.8) `call_uid`
- [ ] Phase 10: Classic RAG Chunking Indexer (`rag-classic/`)
