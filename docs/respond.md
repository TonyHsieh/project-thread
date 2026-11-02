# `respond`

**Module:** [`app/chat.py`](../app/chat.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`respond` builds conversational behavior on top of `generate_text`. It inspects conversation history, verifies caller turns, slices prior context, forwards caller-retrieved evidence, and returns the assistant's reply as a `GenerationResult`.

---

## 2. Function Signature

```python
def respond(
    history: list[Message],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context: list[str] | None = None,
) -> GenerationResult:
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `history` | `list[Message]` | *required* | Full conversation history. Must be non-empty and end with a user turn (`history[-1].role == 'user'`). |
| `system_prompt` | `str` | `"You are a helpful assistant."` | High-level persona and behavioral guidance. |
| `context` | `list[str] \| None` | `None` | Optional rank-ordered list of evidence texts retrieved for this turn. |

### Return Value
Returns [`GenerationResult`](domain_models.md#generationresult).

---

## 3. Invariants & Guarantees
1. **Last Message Role Requirement**:
   The last element of `history` (`history[-1]`) must have `role == "user"`. If `history` is empty or ends with `assistant` or `tool`, a `ValueError` is raised.
2. **Non-Mutation Invariant**:
   `respond` does not modify the caller's `history` list in place. It decomposes `history[-1].content` as the prompt and `history[:-1]` as past turns. Appending user and assistant messages is managed explicitly by the caller (or `chat_loop`).
3. **Seam Decoupling**:
   Retrieval is caller-side. `respond` never opens the corpus or executes searches internally; it merely forwards caller-supplied `context` unchanged to `generate_text`.
4. **Role Separation**:
   `history` (conversation turns), `context` (turn-specific evidence), and `system_prompt` (caller persona) remain strictly distinct.

---

## 4. Usage Example

```python
from domain.models import Message
from chat import respond

conversation = [
    Message(role="user", content="What is an embedding vector?"),
    Message(role="assistant", content="An embedding is a numeric vector representing concept meaning."),
    Message(role="user", content="How do you measure similarity between two embeddings?"),
]

result = respond(
    history=conversation,
    system_prompt="You are an expert AI instructor.",
)

print(result.text)
# Note: conversation list is NOT mutated:
assert len(conversation) == 3
```

