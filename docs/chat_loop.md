# `chat_loop`

**Module:** [`app/chat.py`](../app/chat.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`chat_loop` runs the interactive terminal console adapter for multi-turn conversations. It explicitly manages in-memory conversation history, executes special adapter commands (`/reset`, `/search`), integrates streaming display, and supports optional caller-side grounded retrieval.

---

## 2. Function Signature

```python
def chat_loop(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    stream: bool = True,
    use_search: bool = False,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `system_prompt` | `str` | `"You are a helpful assistant."` | Persona and instructions. |
| `stream` | `bool` | `True` | Stream delta chunks in real time if `True`. |
| `use_search` | `bool` | `False` | Automatically query corpus and inject evidence context if `True`. |
| `input_stream` | `TextIO \| None` | `None` | Defaults to `sys.stdin`; accepts custom stream for automated testing. |
| `output_stream` | `TextIO \| None` | `None` | Defaults to `sys.stdout`; accepts custom stream for automated testing. |

---

## 3. Console Commands

| Command | Action | Description |
| :--- | :--- | :--- |
| `/reset` | Clear history | Replaces in-memory conversation list with an empty list `[]` and prints confirmation. The subsequent turn receives zero facts from prior turns. |
| `/search` or `/rag` | Toggle Grounding | Toggles active corpus retrieval on and off during the live chat session. |
| `/exit`, `/quit`, `/q` | Terminate session | Cleanly breaks out of the loop and exits. |

---

## 4. Execution & CLI Options

Run default pure chat:
```bash
python app/chat.py
```

Run with grounded retrieval preview enabled at startup:
```bash
python app/chat.py --search
# or
python app/chat.py --rag
```

### Sample Session
```text
============================================================
 Project Thread Chat Console (type /reset to clear, /exit to quit)
 [Grounded Retrieval Preview: ACTIVE (type /search to toggle)]
============================================================

You: What is an embedding?
[Grounding: retrieved 1 document(s)]

Assistant: An embedding is a numerical vector representing semantic meaning...

You: /reset
[System] Conversation history reset. Memory cleared.

You: /exit
Exiting session. Goodbye!
```

