# `respond_stream`

**Module:** [`app/chat.py`](../app/chat.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`respond_stream` is the streaming counterpart of [`respond`](respond.md). It verifies conversation history invariants and returns a [`TextStream`](domain_models.md#textstream) for real-time display in interactive chat interfaces.

---

## 2. Function Signature

```python
def respond_stream(
    history: list[Message],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context: list[str] | None = None,
) -> TextStream:
```

### Parameters
Parameters are identical to [`respond`](respond.md):
- `history: list[Message]` (must end with `role == 'user'`)
- `system_prompt: str`
- `context: list[str] | None`

### Return Value
Returns [`TextStream`](domain_models.md#textstream).

---

## 3. Invariants & Guarantees
1. **Identical Contract with `respond`**:
   Validates non-empty history, enforces `history[-1].role == 'user'`, guarantees non-mutation of the caller's list, and forwards `context` unchanged to `generate_text_stream`.
2. **Timing Parity**:
   Streaming alters display timing only. Once consumed, `stream.result` matches the logical output produced by `respond`.

---

## 4. Usage Example

```python
from domain.models import Message
from chat import respond_stream

history = [
    Message(role="user", content="Explain cosine similarity in two sentences.")
]

stream = respond_stream(history)

print("Assistant: ", end="", flush=True)
for chunk in stream:
    print(chunk, end="", flush=True)
print()

# Appending the assistant response explicitly after stream exhaustion:
history.append(Message(role="assistant", content=stream.result.text))
```

