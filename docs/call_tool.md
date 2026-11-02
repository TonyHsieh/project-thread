# `call_tool`

**Module:** [`app/tools.py`](../app/tools.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`call_tool` executes a registered tool using model-supplied arguments under the principle:
> **The model proposes; code disposes.**

Any model-generated `ToolCall` is untrusted input. `call_tool` validates tool presence, verifies argument payload structures against the declared JSON Schema, rejects undeclared properties, executes the code-side handler under least privilege, and returns a structured [`ToolResult`](domain_models.md#toolresult).

---

## 2. Function Signature

```python
def call_tool(
    name: str,
    arguments: dict[str, Any] | Any,
    call_uid: str | None = None,
) -> ToolResult:
```

### Parameters
| Name | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `name` | `str` | *required* | The registered snake_case name of the tool. |
| `arguments` | `dict[str, Any] \| Any` | *required* | Arguments payload provided by the model or caller. |
| `call_uid` | `str \| None` | `None` | Invocation correlation ID. Copied verbatim or auto-minted if omitted. |

### Return Value
Returns [`ToolResult`](domain_models.md#toolresult):
- `call_uid: str`: Correlation ID matching the input `call_uid`.
- `name: str`: Tool name.
- `ok: bool`: `True` if execution succeeded; `False` if validation or execution failed.
- `content: str`: String response for LLM ingestion / display.
- `data: dict[str, Any] | None`: Raw structured data output.
- `error: ToolError | None`: Error details when `ok == False`.

---

## 3. Invariants & Error Containment

1. **Validation Failure is a Result, Not a Crash**:
   Unknown tools, missing/mistyped parameters, and runtime handler errors return `ToolResult.ok = False` with a stable error code. They never crash the calling process.
2. **Matching Rule (Migration v3.8)**:
   `ToolResult.call_uid` is always the `call_uid` of the call it answers. Even on failure paths (`ok = False`), `call_uid` is preserved unchanged.
3. **Least Privilege**:
   Handlers receive only their validated keyword arguments—never the runtime environment or globals.
4. **Stable Error Code Taxonomy**:
   | Code | Condition |
   | :--- | :--- |
   | `UNKNOWN_TOOL` | Tool name is not registered in the system registry. |
   | `TOOL_DISABLED` | Tool is registered but disabled in `tools_config.json`. |
   | `MALFORMED_ARGUMENTS` | Arguments payload is not a valid JSON object dictionary. |
   | `SCHEMA_VALIDATION_ERROR` | Missing required fields, invalid types, or undeclared arguments. |
   | `EXECUTION_ERROR` | Uncaught exception or `ToolExecutionError` raised during handler execution. |
   | `GUARDRAIL_BLOCKED` | Policy or security restriction violation. |

5. **`ToolExecutionError` vs `ToolError` Boundary Translation**:
   - `ToolExecutionError` (`domain/errors.py`) is an internal Python `Exception` raised by handlers on execution failures.
   - `ToolError` (`domain/models.py`) is the serialized data boundary object inside `ToolResult.error`.
   - `call_tool` catches `ToolExecutionError` (and generic exceptions) and translates them into `ToolError(code="EXECUTION_ERROR", message=str(e))` so failures never crash the runtime.

---

## 4. Usage Examples

### CLI Execution
```bash
# Call calculate
python app/tools.py call calculate "expression=25*4+17"

# Call get_current_time
python app/tools.py call get_current_time "{}"

# Call search_corpus
python app/tools.py call search_corpus "query=tokens top_k=2"
```

### Python Example
```python
import tools

# Successful execution
res = tools.call_tool(
    name="calculate",
    arguments={"expression": "12 * (4 + 3)"},
    call_uid="call_test_01",
)
assert res.ok is True
assert res.data["result"] == 84
assert res.call_uid == "call_test_01"

# Safe error containment (division by zero)
res_err = tools.call_tool(
    name="calculate",
    arguments={"expression": "10 / 0"},
    call_uid="call_test_02",
)
assert res_err.ok is False
assert res_err.error.code == tools.CODE_EXECUTION_ERROR
assert res_err.call_uid == "call_test_02"
```

