"""
Unit and conformance tests for app/tools.py and Tool API (Lesson 11 / Phase 12).

Covers:
- Conformance: The model proposes; code disposes (untrusted model input validation)
- Matching rule: ToolResult.call_uid matches ToolCall.call_uid
- Conformance: Two parallel calls have distinct, matching call_uids
- Error containment: Validation failures (unknown tool, schema violation, malformed payload, handler crash)
  return ToolResult.ok = False with preserved call_uid and stable error codes
- Undeclared argument rejection
- Reference starter tools execution (calculate, search_corpus, get_current_time)
- OpenAI tool schema export (handler never serialized)
"""

from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure app directory is in sys.path
_app_dir = Path(__file__).resolve().parent.parent.parent / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

from domain.models import SearchResult, ToolDefinition, ToolResult
import tools


class TestToolValidationAndExecution(unittest.TestCase):
    """Test validation boundaries and safe execution contracts."""

    def test_unknown_tool_returns_ok_false_with_preserved_uid(self):
        uid = "call_abc123"
        result = tools.call_tool("non_existent_tool", {}, call_uid=uid)

        self.assertIsInstance(result, ToolResult)
        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_UNKNOWN_TOOL)
        self.assertIn("non_existent_tool", result.content)

    def test_malformed_arguments_payload(self):
        uid = "call_malformed"
        result = tools.call_tool("calculate", "not a dictionary payload", call_uid=uid)

        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_MALFORMED_ARGUMENTS)

    def test_schema_validation_missing_required_property(self):
        uid = "call_missing_req"
        # calculate requires "expression"
        result = tools.call_tool("calculate", {}, call_uid=uid)

        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_SCHEMA_VALIDATION_ERROR)
        self.assertIn("expression", result.error.message)

    def test_schema_validation_incorrect_property_type(self):
        uid = "call_wrong_type"
        # search_corpus requires query to be string, not int
        result = tools.call_tool("search_corpus", {"query": 12345}, call_uid=uid)

        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_SCHEMA_VALIDATION_ERROR)

    def test_undeclared_arguments_rejected(self):
        uid = "call_extra_arg"
        # calculate has additionalProperties: False
        result = tools.call_tool(
            "calculate",
            {"expression": "2 + 2", "unexpected_arg": "value"},
            call_uid=uid,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_SCHEMA_VALIDATION_ERROR)
        self.assertIn("unexpected_arg", result.error.message)

    def test_handler_runtime_exception_captured_safely(self):
        uid = "call_div_zero"
        # Division by zero inside calculate
        result = tools.call_tool("calculate", {"expression": "10 / 0"}, call_uid=uid)

        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_EXECUTION_ERROR)
        self.assertIn("division by zero", result.error.message.lower())

    def test_handler_tool_execution_error_captured_safely(self):
        uid = "call_tee_test"
        # Invalid syntax in calculate raises ToolExecutionError internally
        result = tools.call_tool("calculate", {"expression": "invalid syntax ++ *"}, call_uid=uid)

        self.assertFalse(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.code, tools.CODE_EXECUTION_ERROR)
        self.assertIn("Invalid mathematical expression", result.error.message)


class TestStarterToolsExecution(unittest.TestCase):
    """Test standard starter tools functionality."""

    def test_successful_calculate(self):
        uid = "call_calc_success"
        result = tools.call_tool("calculate", {"expression": "12 * (4 + 3)"}, call_uid=uid)

        self.assertTrue(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIsNone(result.error)
        self.assertEqual(result.data["result"], 84)

    def test_successful_get_current_time(self):
        uid = "call_time"
        result = tools.call_tool("get_current_time", {}, call_uid=uid)

        self.assertTrue(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertIn("timestamp_iso", result.data)
        self.assertEqual(result.data["timezone"], "UTC")

    @patch("search.search_corpus")
    def test_successful_search_corpus(self, mock_search):
        mock_search.return_value = [
            SearchResult(
                source_filename="doc1.txt",
                chunk_id="doc1.txt:0",
                text="Artificial Intelligence concepts",
                score=0.92,
                metadata={"subdirectory": ""},
            )
        ]
        uid = "call_search_1"
        result = tools.call_tool("search_corpus", {"query": "AI concepts", "top_k": 3}, call_uid=uid)

        self.assertTrue(result.ok)
        self.assertEqual(result.call_uid, uid)
        self.assertEqual(result.data["count"], 1)
        self.assertEqual(result.data["results"][0]["source_filename"], "doc1.txt")


class TestMatchingAndRegistryRules(unittest.TestCase):
    """Test matching rule, correlation, and registry invariants."""

    def test_conformance_parallel_calls_distinct_call_uids(self):
        """
        Conformance check: two parallel calls to the same tool return two results
        whose call_uids are distinct and match their respective ToolCall.call_uid.
        """
        uid1 = "call_req_001"
        uid2 = "call_req_002"

        res1 = tools.call_tool("calculate", {"expression": "1 + 1"}, call_uid=uid1)
        res2 = tools.call_tool("calculate", {"expression": "2 + 2"}, call_uid=uid2)

        self.assertEqual(res1.call_uid, uid1)
        self.assertEqual(res2.call_uid, uid2)
        self.assertNotEqual(res1.call_uid, res2.call_uid)
        self.assertEqual(res1.data["result"], 2)
        self.assertEqual(res2.data["result"], 4)

    def test_auto_minted_call_uid_when_omitted(self):
        result = tools.call_tool("calculate", {"expression": "5 * 5"})
        self.assertTrue(result.ok)
        self.assertTrue(result.call_uid.startswith("call_"))

    def test_get_openai_tools_excludes_handler(self):
        openai_specs = tools.get_openai_tools()
        self.assertGreaterEqual(len(openai_specs), 3)

        for spec in openai_specs:
            self.assertEqual(spec["type"], "function")
            fn = spec["function"]
            self.assertIn("name", fn)
            self.assertIn("description", fn)
            self.assertIn("parameters", fn)
            # Security guarantee: code-side handler is NEVER serialized
            self.assertNotIn("handler", fn)
            self.assertNotIn("handler", spec)

    def test_registry_enforces_snake_case_and_uniqueness(self):
        custom_registry = tools.ToolRegistry()

        valid_tool = ToolDefinition(
            name="valid_tool_name",
            description="A valid tool",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: "ok",
        )
        custom_registry.register(valid_tool)

        # Duplicate registration raises ValueError
        with self.assertRaises(ValueError):
            custom_registry.register(valid_tool)

        # Non-snake_case raises ValueError
        invalid_name_tool = ToolDefinition(
            name="InvalidToolName",
            description="Bad name",
            input_schema={"type": "object"},
            handler=lambda: "ok",
        )
        with self.assertRaises(ValueError):
            custom_registry.register(invalid_name_tool)

    def test_tool_enable_disable_lifecycle(self):
        # Disable calculate
        tools.disable_tool("calculate")
        self.assertFalse(tools.is_tool_enabled("calculate"))

        # Calling disabled tool returns ok=False with CODE_TOOL_DISABLED
        res = tools.call_tool("calculate", {"expression": "1 + 1"}, call_uid="call_dis_1")
        self.assertFalse(res.ok)
        self.assertEqual(res.call_uid, "call_dis_1")
        self.assertEqual(res.error.code, tools.CODE_TOOL_DISABLED)

        # Disabled tool omitted from OpenAI export
        openai_tools = tools.get_openai_tools()
        tool_names = [t["function"]["name"] for t in openai_tools]
        self.assertNotIn("calculate", tool_names)

        # Re-enable calculate
        tools.enable_tool("calculate")
        self.assertTrue(tools.is_tool_enabled("calculate"))
        res_after = tools.call_tool("calculate", {"expression": "1 + 1"}, call_uid="call_en_1")
        self.assertTrue(res_after.ok)
        self.assertEqual(res_after.data["result"], 2)


if __name__ == "__main__":
    unittest.main()

