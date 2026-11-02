# `get_openai_tools`

**Module:** [`app/tools.py`](../app/tools.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`get_openai_tools` exports all registered and currently enabled tools into the standard OpenAI Function Calling specification. It safely strips Python `handler` callables and honors runtime configuration in [`app/tools_config.json`](../app/tools_config.json).

---

## 2. Function Signature

```python
def get_openai_tools() -> list[dict[str, Any]]:
```

### Return Value
Returns a list of dictionaries matching the OpenAI Tools format:
```json
[
  {
    "type": "function",
    "function": {
      "name": "<tool_name>",
      "description": "<tool_description>",
      "parameters": { ...JSON Schema... }
    }
  }
]
```

---

## 3. Invariants & Security Guarantees
1. **Handler Isolation**:
   Python `handler` callables are code-side only. `get_openai_tools()` strictly exports only `name`, `description`, and `parameters`.
2. **Persistent Configuration Filtering**:
   Tools marked disabled in `tools_config.json` (or via `python app/tools.py disable <name>`) are automatically excluded from the returned list.

---

## 4. Usage Example

```python
import tools
import config

# Retrieve registered tools for LLM invocation
openai_tools = tools.get_openai_tools()

# Dispatch to model via OpenAI client
client = config.get_client()
response = client.chat.completions.create(
    model=config.get_default_model_id(),
    messages=[{"role": "user", "content": "What is 25 * 4 + 17?"}],
    tools=openai_tools,
)

# Inspect model proposed tool calls
choice = response.choices[0]
if choice.message.tool_calls:
    for call in choice.message.tool_calls:
        print(f"Proposed tool: {call.function.name}")
        print(f"Arguments: {call.function.arguments}")
```

