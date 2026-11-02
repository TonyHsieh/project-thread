# `register_tool` & `ToolRegistry`

**Module:** [`app/tools.py`](../app/tools.py)  
**Contract Version:** `project-thread-api/v1`

---

## 1. Overview
`register_tool` registers a [`ToolDefinition`](domain_models.md#tooldefinition) into the application's central `ToolRegistry`. This provides a single source of truth across chat adapters, multi-step agents, MCP servers, and HTTP endpoints, preventing schema divergence.

---

## 2. Function Signature

```python
def register_tool(definition: ToolDefinition) -> None:
```

### Parameters
| Name | Type | Description |
| :--- | :--- | :--- |
| `definition` | [`ToolDefinition`](domain_models.md#tooldefinition) | Immutable definition holding `name`, `description`, `input_schema`, and Python `handler`. |

---

## 3. Invariants & Registration Rules
1. **Naming**: Must be snake_case (matching `^[a-z][a-z0-9_]*$`).
2. **Uniqueness**: Tool names must be unique within the registry. Registering a duplicate name raises `ValueError`.
3. **JSON Schema**: `input_schema` must be a valid dictionary with `type: "object"`.
4. **Code-Side Handler**: The callable `handler` is executed only by `call_tool`. It is never serialized or transmitted to the model backend.

---

## 4. Usage Example

```python
from domain.models import ToolDefinition
import tools

# Define a custom tool
def roll_dice_handler(sides: int = 6) -> dict:
    import random
    return {"sides": sides, "roll": random.randint(1, sides)}

dice_tool = ToolDefinition(
    name="roll_dice",
    description="Roll a virtual die with a specified number of sides.",
    input_schema={
        "type": "object",
        "properties": {
            "sides": {
                "type": "integer",
                "description": "Number of sides on the die (default 6).",
                "default": 6,
                "minimum": 2,
            }
        },
        "additionalProperties": False,
    },
    handler=roll_dice_handler,
)

# Register tool
tools.register_tool(dice_tool)

# Execute via call_tool
result = tools.call_tool("roll_dice", {"sides": 20})
print(f"Result: {result.data['roll']}")
```

