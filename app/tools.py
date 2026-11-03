"""
Tool API and Function-Calling Registry for project-thread-api/v1 (Lesson 11 / Phase 12).

Provides:
- ToolDefinition: Declaration of tools with strict JSON Schema.
- ToolRegistry: Single source of truth for tool declarations.
- call_tool(name, arguments, call_uid): Model proposes; code disposes.
  Validates arguments against JSON schema, applies least-privilege execution,
  and returns ToolResult.ok = False on validation failure (never crashing).
- Conformance to Migration v3.8 (call_uid correlation).
"""

import ast
from collections.abc import Callable
from datetime import datetime, timezone
import json
import operator
from pathlib import Path
import re
import sys
from typing import Any
import uuid

import jsonschema

from domain.errors import ToolExecutionError
from domain.models import SearchResult, ToolDefinition, ToolError, ToolResult
import search

# ---------------------------------------------------------------------------
# Stable Error Codes
# ---------------------------------------------------------------------------
CODE_UNKNOWN_TOOL = "UNKNOWN_TOOL"
CODE_MALFORMED_ARGUMENTS = "MALFORMED_ARGUMENTS"
CODE_SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
CODE_EXECUTION_ERROR = "EXECUTION_ERROR"
CODE_GUARDRAIL_BLOCKED = "GUARDRAIL_BLOCKED"
CODE_TOOL_DISABLED = "TOOL_DISABLED"


# ---------------------------------------------------------------------------
# Tool Configuration Persistence (tools_config.json)
# ---------------------------------------------------------------------------
def get_config_path() -> Path:
    return Path(__file__).resolve().parent / "tools_config.json"


def load_tools_config() -> dict[str, Any]:
    """Load persisted tool configuration or return defaults."""
    path = get_config_path()
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"disabled_tools": []}
    return {"disabled_tools": []}


def save_tools_config(config_data: dict[str, Any]) -> None:
    """Save tool configuration to tools_config.json."""
    path = get_config_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
    except Exception as e:
        pass


def is_tool_enabled(name: str) -> bool:
    """Check if a tool is enabled in persistent configuration."""
    cfg = load_tools_config()
    disabled_list = cfg.get("disabled_tools", [])
    return name not in disabled_list


def enable_tool(name: str) -> bool:
    """Enable a tool by removing it from disabled_tools in configuration."""
    cfg = load_tools_config()
    disabled = cfg.get("disabled_tools", [])
    if name in disabled:
        disabled.remove(name)
        cfg["disabled_tools"] = disabled
        save_tools_config(cfg)
    return True


def disable_tool(name: str) -> bool:
    """Disable a tool by recording it in disabled_tools in configuration."""
    cfg = load_tools_config()
    disabled = cfg.get("disabled_tools", [])
    if name not in disabled:
        disabled.append(name)
        cfg["disabled_tools"] = disabled
        save_tools_config(cfg)
    return True


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """Registry maintaining unique ToolDefinitions for the application."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        """Register a ToolDefinition once. Name must be snake_case and unique."""
        if not re.match(r"^[a-z][a-z0-9_]*$", definition.name):
            raise ValueError(
                f"Tool name '{definition.name}' must be snake_case (e.g. 'search_corpus')."
            )
        if definition.name in self._tools:
            raise ValueError(f"Tool '{definition.name}' is already registered.")

        if not isinstance(definition.input_schema, dict):
            raise TypeError("Tool input_schema must be a dictionary representing a JSON Schema.")

        if definition.input_schema.get("type") != "object":
            raise ValueError("Tool input_schema must have type 'object'.")

        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """
        Format registered and enabled tools for OpenAI function calling.
        Handler callable is code-side only and never serialized to the model.
        """
        openai_tools: list[dict[str, Any]] = []
        for tool in self._tools.values():
            if not is_tool_enabled(tool.name):
                continue
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
            )
        return openai_tools


# Global registry instance
registry = ToolRegistry()


# ---------------------------------------------------------------------------
# Tool Execution Boundary: call_tool
# ---------------------------------------------------------------------------
def call_tool(
    name: str,
    arguments: dict[str, Any] | Any,
    call_uid: str | None = None,
) -> ToolResult:
    """
    Execute a tool by name with model-supplied arguments under least privilege.

    Contract guarantees:
    - The model proposes; code disposes. Arguments are validated against JSON schema.
    - Validation failure is a result (ToolResult.ok = False), not a crash.
    - call_uid is preserved unchanged on both success and failure paths.
    - Handlers receive only their validated keyword arguments.
    """
    # Matching rule: copy call_uid verbatim or mint a unique uid before dispatch
    effective_uid = call_uid if (call_uid is not None and str(call_uid).strip()) else f"call_{uuid.uuid4().hex[:12]}"

    # 1. Lookup tool in registry
    tool = registry.get(name)
    if tool is None:
        return ToolResult(
            call_uid=effective_uid,
            name=name,
            ok=False,
            content=f"Unknown tool '{name}'. Tool is not registered in the system registry.",
            error=ToolError(
                code=CODE_UNKNOWN_TOOL,
                message=f"Tool '{name}' is not registered.",
            ),
        )

    # 2. Check if tool is disabled in configuration
    if not is_tool_enabled(name):
        return ToolResult(
            call_uid=effective_uid,
            name=name,
            ok=False,
            content=f"Tool '{name}' is currently disabled in tools configuration.",
            error=ToolError(
                code=CODE_TOOL_DISABLED,
                message=f"Tool '{name}' is disabled in tools_config.json.",
            ),
        )

    # 3. Validate arguments payload is a dictionary
    if not isinstance(arguments, dict):
        return ToolResult(
            call_uid=effective_uid,
            name=name,
            ok=False,
            content="Malformed arguments payload: arguments must be a JSON dictionary object.",
            error=ToolError(
                code=CODE_MALFORMED_ARGUMENTS,
                message="Arguments payload must be a JSON object dictionary.",
            ),
        )

    # 3. Validate arguments against JSON Schema
    try:
        jsonschema.validate(instance=arguments, schema=tool.input_schema)
    except jsonschema.exceptions.ValidationError as e:
        return ToolResult(
            call_uid=effective_uid,
            name=name,
            ok=False,
            content=f"Schema validation failed: {e.message}",
            error=ToolError(
                code=CODE_SCHEMA_VALIDATION_ERROR,
                message=e.message,
            ),
        )

    # 4. Strict check: forbid undeclared properties by default unless additionalProperties: True
    if tool.input_schema.get("type") == "object":
        declared_properties = set(tool.input_schema.get("properties", {}).keys())
        allow_additional = tool.input_schema.get("additionalProperties", False)
        if not allow_additional:
            extra_keys = set(arguments.keys()) - declared_properties
            if extra_keys:
                sorted_extras = sorted(list(extra_keys))
                msg = f"Undeclared argument(s) passed: {', '.join(sorted_extras)}."
                return ToolResult(
                    call_uid=effective_uid,
                    name=name,
                    ok=False,
                    content=f"Schema validation failed: {msg}",
                    error=ToolError(
                        code=CODE_SCHEMA_VALIDATION_ERROR,
                        message=msg,
                    ),
                )

    # 5. Execute handler under least privilege
    try:
        raw_output = tool.handler(**arguments)
    except ToolExecutionError as e:
        return ToolResult(
            call_uid=effective_uid,
            name=name,
            ok=False,
            content=f"Tool '{name}' execution error: {e}",
            error=ToolError(
                code=CODE_EXECUTION_ERROR,
                message=str(e),
            ),
        )
    except Exception as e:
        return ToolResult(
            call_uid=effective_uid,
            name=name,
            ok=False,
            content=f"Tool '{name}' unexpected execution failure: {e}",
            error=ToolError(
                code=CODE_EXECUTION_ERROR,
                message=str(e),
            ),
        )

    # 6. Format successful output and extract sources if present
    extracted_sources: tuple[SearchResult, ...] | None = None
    if isinstance(raw_output, dict):
        if "sources" in raw_output and isinstance(raw_output["sources"], (list, tuple)):
            found_sources = tuple(s for s in raw_output["sources"] if isinstance(s, SearchResult))
            if found_sources:
                extracted_sources = found_sources
        content = json.dumps(raw_output, default=str)
        data = raw_output
    elif isinstance(raw_output, (list, tuple)):
        found_sources = tuple(s for s in raw_output if isinstance(s, SearchResult))
        if found_sources:
            extracted_sources = found_sources
        content = json.dumps(raw_output, default=str)
        data = {"items": raw_output}
    elif isinstance(raw_output, str):
        content = raw_output
        data = None
    else:
        content = str(raw_output)
        data = {"value": raw_output}

    return ToolResult(
        call_uid=effective_uid,
        name=name,
        ok=True,
        content=content,
        data=data,
        error=None,
        sources=extracted_sources,
    )


# ---------------------------------------------------------------------------
# Reference Starter Tools
# ---------------------------------------------------------------------------

# Safe AST Math Evaluator
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_math(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _safe_eval_math(node.left)
        right = _safe_eval_math(node.right)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval_math(node.operand)
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate_handler(expression: str) -> dict[str, Any]:
    """Evaluate a mathematical expression securely via AST parsing."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval_math(tree.body)
        return {"expression": expression, "result": result}
    except Exception as e:
        raise ToolExecutionError(f"Invalid mathematical expression '{expression}': {e}") from e


def search_corpus_handler(query: str, top_k: int = 5) -> dict[str, Any]:
    """Perform corpus search using Phase 1 search_corpus."""
    results = search.search_corpus(query=query, top_k=top_k)
    serialized = [
        {
            "source_filename": r.source_filename,
            "chunk_id": r.chunk_id,
            "score": r.score,
            "text": r.text,
            "metadata": r.metadata,
        }
        for r in results
    ]
    return {
        "query": query,
        "results": serialized,
        "count": len(serialized),
        "sources": results,
    }


def get_current_time_handler(timezone_name: str = "UTC") -> dict[str, Any]:
    """Return current timestamp in ISO 8601 format."""
    now = datetime.now(timezone.utc)
    return {
        "timestamp_iso": now.isoformat(),
        "utc_date": now.strftime("%Y-%m-%d"),
        "utc_time": now.strftime("%H:%M:%S UTC"),
        "timezone": timezone_name,
    }


# Register reference starter tools
registry.register(
    ToolDefinition(
        name="search_corpus",
        description="Search the knowledge corpus for relevant documents and evidence using cosine similarity.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search text or question to look up in the corpus.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (default 5).",
                    "default": 5,
                    "minimum": 1,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=search_corpus_handler,
    )
)

registry.register(
    ToolDefinition(
        name="calculate",
        description="Safely evaluate a mathematical expression (supports +, -, *, /, %, **).",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate, e.g. '12 * (4 + 3)'.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=calculate_handler,
    )
)

registry.register(
    ToolDefinition(
        name="get_current_time",
        description="Get the current date and time in UTC.",
        input_schema={
            "type": "object",
            "properties": {
                "timezone_name": {
                    "type": "string",
                    "description": "Name of timezone, default 'UTC'.",
                    "default": "UTC",
                },
            },
            "additionalProperties": False,
        },
        handler=get_current_time_handler,
    )
)


def register_tool(definition: ToolDefinition) -> None:
    """Convenience function to register a tool in the default registry."""
    registry.register(definition)


def get_openai_tools() -> list[dict[str, Any]]:
    """Convenience function to export OpenAI function-calling specifications."""
    return registry.get_openai_tools()


def main() -> None:
    """Standalone CLI test runner and configuration manager for tools.py."""
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print("============================================================")
        print(" Project Thread Tool API & Function Registry (app/tools.py)")
        print("============================================================")
        print("Usage:")
        print("  python app/tools.py list                       - List all registered tools and their status")
        print("  python app/tools.py enable <tool_name>         - Enable a tool in tools_config.json")
        print("  python app/tools.py disable <tool_name>        - Disable a tool in tools_config.json")
        print("  python app/tools.py call <name> '<json_args>'  - Execute a tool directly with JSON arguments")
        print("  python app/tools.py test                       - Run quick built-in test suite on all registered tools")
        print("\nRegistered Tools Overview:")
        for tool in registry.list_all():
            status = "ENABLED" if is_tool_enabled(tool.name) else "DISABLED"
            print(f"  • {tool.name:<18} [{status}] : {tool.description}")
        return

    cmd = args[0].lower()
    match cmd:
        case "list":
            print("\nRegistered Tools in Registry:")
            print("-" * 65)
            for tool in registry.list_all():
                status = "ENABLED" if is_tool_enabled(tool.name) else "DISABLED"
                print(f"Name       : {tool.name}")
                print(f"Status     : {status}")
                print(f"Description: {tool.description}")
                print(f"Schema     : {json.dumps(tool.input_schema, indent=2)}")
                print("-" * 65)

        case "enable":
            if len(args) < 2:
                print("Error: Specify tool name to enable (e.g. 'python app/tools.py enable calculate')")
                return
            tool_name = args[1]
            if registry.get(tool_name) is None:
                print(f"Warning: Tool '{tool_name}' is not in the registry, but updating configuration.")
            enable_tool(tool_name)
            print(f"[Success] Tool '{tool_name}' enabled in {get_config_path().name}")

        case "disable":
            if len(args) < 2:
                print("Error: Specify tool name to disable (e.g. 'python app/tools.py disable calculate')")
                return
            tool_name = args[1]
            disable_tool(tool_name)
            print(f"[Success] Tool '{tool_name}' disabled in {get_config_path().name}")

        case "call":
            if len(args) < 2:
                print("Error: Specify tool name to call (e.g. 'python app/tools.py call get_current_time')")
                return
            tool_name = args[1]
            payload_raw = args[2] if len(args) > 2 else "{}"
            if payload_raw.startswith("{"):
                try:
                    payload = json.loads(payload_raw)
                except json.JSONDecodeError:
                    try:
                        payload = ast.literal_eval(payload_raw)
                        if not isinstance(payload, dict):
                            raise ValueError()
                    except Exception as e:
                        print(f"Error: Invalid JSON dictionary payload: {e}")
                        return
            else:
                payload = {}
                for item in args[2:]:
                    if "=" in item:
                        k, v = item.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        try:
                            v = int(v)
                        except ValueError:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                        payload[k] = v
                    else:
                        print(f"Error: Unrecognized argument format '{item}'. Use key=value or a JSON string.")
                        return

            print(f"\n--- Calling tool '{tool_name}' with arguments: {payload} ---")
            result = call_tool(tool_name, payload)
            print(f"Status:   {'OK' if result.ok else 'FAILED'}")
            print(f"Call UID: {result.call_uid}")
            if result.ok:
                print(f"Content:  {result.content}")
                if result.data:
                    print(f"Data:     {json.dumps(result.data, indent=2)}")
            else:
                print(f"Error:    [{result.error.code}] {result.error.message}")

        case "test":
            print("\n--- Running standalone self-test on registered tools ---")
            # 1. calculate test
            calc_res = call_tool("calculate", {"expression": "25 * 4 + 17"})
            calc_val = calc_res.data.get("result") if calc_res.data else None
            print(f"1. calculate ('25 * 4 + 17'): ok={calc_res.ok}, result={calc_val}")

            # 2. get_current_time test
            time_res = call_tool("get_current_time", {})
            time_val = time_res.data.get("timestamp_iso") if time_res.data else None
            print(f"2. get_current_time: ok={time_res.ok}, timestamp={time_val}")

            # 3. search_corpus test
            search_res = call_tool("search_corpus", {"query": "tokens", "top_k": 2})
            search_cnt = search_res.data.get("count") if search_res.data else None
            print(f"3. search_corpus ('tokens'): ok={search_res.ok}, count={search_cnt}")
            print("\nSelf-test completed successfully.")

        case _:
            print(f"Unknown command: '{cmd}'. Run 'python app/tools.py --help' for usage.")


if __name__ == "__main__":
    main()

