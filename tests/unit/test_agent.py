"""
Unit and boundary tests for app/agent.py (Phase 13: Autonomous Agent Loop).

Covers:
- Positive path: single-step direct answer without tools
- Positive path: tool invocation cycle (Decide -> call_tool -> Observe -> Final answer)
- Negative path: input validation (empty prompt, invalid max_steps)
- Boundary condition: max_steps exhaustion (finish_reason='max_steps')
- Edge case: tool execution error containment and recovery in observation
- Edge case: search_corpus sources accumulation
- Failure modes: ModelError handling (finish_reason='model_error')
- Usage accumulation and adherence to The Null Rule
- Assistant tool_calls serialization and round 2 message sequence in textgen
"""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Ensure app directory is in sys.path
_app_dir = Path(__file__).resolve().parent.parent.parent / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import agent
from domain.errors import ModelError
from domain.models import (
    AgentResult,
    GenerationResult,
    Message,
    SearchResult,
    ToolCall,
    ToolResult,
    Usage,
)
import textgen
import tools


class TestAgentValidation(unittest.TestCase):
    """Test parameter validation for run_agent."""

    def test_empty_prompt_raises(self):
        with self.assertRaises(ValueError):
            agent.run_agent("")
        with self.assertRaises(ValueError):
            agent.run_agent("   \n\t  ")

    def test_invalid_max_steps_raises(self):
        with self.assertRaises(ValueError):
            agent.run_agent("Hello", max_steps=0)
        with self.assertRaises(ValueError):
            agent.run_agent("Hello", max_steps=-2)
        with self.assertRaises(ValueError):
            agent.run_agent("Hello", max_steps="5")  # type: ignore


class TestAgentExecution(unittest.TestCase):
    """Test autonomous loop decision cycles, tool execution, and state tracking."""

    @patch("textgen.generate_text")
    def test_single_step_direct_answer(self, mock_gen):
        mock_gen.return_value = GenerationResult(
            text="Hello there!",
            model_id="gpt-4o",
            tokens_in=10,
            tokens_out=5,
            est_cost=None,
            finish_reason="stop",
            tool_calls=None,
        )

        res = agent.run_agent("Hi", max_steps=5)

        self.assertIsInstance(res, AgentResult)
        self.assertEqual(res.answer, "Hello there!")
        self.assertEqual(res.finish_reason, "final")
        self.assertEqual(res.steps, 1)
        self.assertEqual(len(res.tool_calls), 0)
        self.assertEqual(len(res.sources), 0)
        self.assertTrue(res.trace_id.startswith("trace_"))
        self.assertEqual(res.usage.tokens_in, 10)
        self.assertEqual(res.usage.tokens_out, 5)
        self.assertIsNone(res.usage.est_cost)

    @patch("textgen.generate_text")
    def test_tool_invocation_cycle(self, mock_gen):
        # Round 1: Model requests calculate(expression="15 * 4 + 7")
        call_uid = "call_test_calc_123"
        round_1_result = GenerationResult(
            text="",
            model_id="gpt-4o",
            tokens_in=20,
            tokens_out=10,
            est_cost=None,
            finish_reason="tool_calls",
            tool_calls=(
                ToolCall(
                    call_uid=call_uid,
                    name="calculate",
                    arguments={"expression": "15 * 4 + 7"},
                ),
            ),
        )

        # Round 2: Model receives observation "67" and provides final answer
        round_2_result = GenerationResult(
            text="The result of 15 * 4 + 7 is 67.",
            model_id="gpt-4o",
            tokens_in=35,
            tokens_out=12,
            est_cost=None,
            finish_reason="stop",
            tool_calls=None,
        )

        mock_gen.side_effect = [round_1_result, round_2_result]

        res = agent.run_agent("What is 15 * 4 + 7?", max_steps=5)

        self.assertEqual(res.finish_reason, "final")
        self.assertEqual(res.answer, "The result of 15 * 4 + 7 is 67.")
        self.assertEqual(res.steps, 2)
        self.assertEqual(len(res.tool_calls), 1)
        self.assertEqual(res.tool_calls[0].name, "calculate")
        self.assertEqual(res.tool_calls[0].call_uid, call_uid)

        # Verify usage accumulated across rounds
        self.assertEqual(res.usage.tokens_in, 55)
        self.assertEqual(res.usage.tokens_out, 22)

        # Verify the message history sent on Round 2
        second_call_kwargs = mock_gen.call_args_list[1][1]
        history = second_call_kwargs["history"]
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].tool_calls[0].call_uid, call_uid)
        self.assertEqual(history[2].role, "tool")
        self.assertEqual(history[2].call_uid, call_uid)
        self.assertIn("67", history[2].content)

    @patch("tools.call_tool")
    @patch("textgen.generate_text")
    def test_sources_accumulation_from_search(self, mock_gen, mock_call_tool):
        call_uid = "call_search_1"
        mock_gen.side_effect = [
            GenerationResult(
                text="",
                model_id="gpt-4o",
                tokens_in=10,
                tokens_out=5,
                est_cost=None,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        call_uid=call_uid,
                        name="search_corpus",
                        arguments={"query": "transformers"},
                    ),
                ),
            ),
            GenerationResult(
                text="Transformers rely on self-attention.",
                model_id="gpt-4o",
                tokens_in=30,
                tokens_out=8,
                est_cost=None,
                finish_reason="stop",
            ),
        ]

        mock_call_tool.return_value = ToolResult(
            call_uid=call_uid,
            name="search_corpus",
            ok=True,
            content="[1] (doc1.txt) Self-attention mechanism...",
            data={
                "query": "transformers",
                "count": 1,
                "results": [
                    {
                        "source_filename": "doc1.txt",
                        "chunk_id": "doc1.txt:0",
                        "score": 0.92,
                        "text": "Self-attention mechanism...",
                        "metadata": {"ext": ".txt"},
                    }
                ],
            },
        )

        res = agent.run_agent("Tell me about transformers", max_steps=5)

        self.assertEqual(res.finish_reason, "final")
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0].source_filename, "doc1.txt")
        self.assertEqual(res.sources[0].score, 0.92)

    @patch("tools.call_tool")
    @patch("textgen.generate_text")
    def test_sources_accumulation_generic_tool(self, mock_gen, mock_call_tool):
        """Verify sources are accumulated based on SearchResult type, regardless of tool name."""
        call_uid = "call_custom_retriever"
        mock_gen.side_effect = [
            GenerationResult(
                text="",
                model_id="gpt-4o",
                tokens_in=10,
                tokens_out=5,
                est_cost=None,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        call_uid=call_uid,
                        name="arbitrary_vector_retriever",
                        arguments={"query": "agentic patterns"},
                    ),
                ),
            ),
            GenerationResult(
                text="Patterns include ReAct and orchestrators.",
                model_id="gpt-4o",
                tokens_in=20,
                tokens_out=6,
                est_cost=None,
                finish_reason="stop",
            ),
        ]

        expected_source = SearchResult(
            source_filename="patterns.md",
            chunk_id="patterns.md:1",
            text="ReAct pattern details...",
            score=0.95,
            metadata={"source": "custom"},
        )

        mock_call_tool.return_value = ToolResult(
            call_uid=call_uid,
            name="arbitrary_vector_retriever",
            ok=True,
            content="Retrieved patterns.md",
            sources=(expected_source,),
        )

        res = agent.run_agent("Find patterns", max_steps=5)

        self.assertEqual(res.finish_reason, "final")
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0].source_filename, "patterns.md")
        self.assertEqual(res.sources[0].score, 0.95)

    @patch("textgen.generate_text")
    def test_tool_error_fed_back_and_recovered(self, mock_gen):
        call_uid = "call_err_1"
        mock_gen.side_effect = [
            # Round 1: Model calls nonexistent tool
            GenerationResult(
                text="",
                model_id="gpt-4o",
                tokens_in=10,
                tokens_out=5,
                est_cost=None,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        call_uid=call_uid,
                        name="non_existent_tool",
                        arguments={},
                    ),
                ),
            ),
            # Round 2: Model recognizes error and answers without crashing
            GenerationResult(
                text="I could not run that tool, but I can answer directly.",
                model_id="gpt-4o",
                tokens_in=25,
                tokens_out=15,
                est_cost=None,
                finish_reason="stop",
            ),
        ]

        res = agent.run_agent("Run non-existent tool", max_steps=5)

        self.assertEqual(res.finish_reason, "final")
        self.assertEqual(res.steps, 2)
        second_history = mock_gen.call_args_list[1][1]["history"]
        tool_msg = second_history[2]
        self.assertEqual(tool_msg.role, "tool")
        self.assertIn("UNKNOWN_TOOL", tool_msg.content)

    @patch("textgen.generate_text")
    def test_max_steps_exhaustion(self, mock_gen):
        # Model perpetually calls tool
        mock_gen.return_value = GenerationResult(
            text="Still thinking...",
            model_id="gpt-4o",
            tokens_in=10,
            tokens_out=5,
            est_cost=None,
            finish_reason="tool_calls",
            tool_calls=(
                ToolCall(
                    call_uid="loop_call",
                    name="get_current_time",
                    arguments={},
                ),
            ),
        )

        res = agent.run_agent("Perpetual loop test", max_steps=3)

        self.assertEqual(res.finish_reason, "max_steps")
        self.assertEqual(res.steps, 3)
        self.assertEqual(len(res.tool_calls), 3)
        self.assertEqual(res.answer, "Still thinking...")

    @patch("textgen.generate_text")
    def test_model_error_handling(self, mock_gen):
        mock_gen.side_effect = ModelError("Upstream API connection timeout")

        res = agent.run_agent("Test error handling", max_steps=5)

        self.assertEqual(res.finish_reason, "model_error")
        self.assertEqual(res.steps, 1)
        self.assertIn("Upstream API connection timeout", res.answer)


class TestTextgenToolExtensions(unittest.TestCase):
    """Test textgen message serialization with assistant tool calls."""

    def test_assistant_tool_calls_serialization(self):
        call = ToolCall(
            call_uid="c_123",
            name="calculate",
            arguments={"expression": "100 / 4"},
        )
        msg_assistant = Message(
            role="assistant",
            content="",
            tool_calls=(call,),
        )
        msg_tool = Message(
            role="tool",
            name="calculate",
            content="25",
            call_uid="c_123",
        )
        history = [msg_assistant, msg_tool]

        messages = textgen._build_messages(
            prompt=None,
            system_prompt="Test system",
            history=history,
            context=None,
        )

        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertIn("tool_calls", messages[1])
        tc_dict = messages[1]["tool_calls"][0]
        self.assertEqual(tc_dict["id"], "c_123")
        self.assertEqual(tc_dict["function"]["name"], "calculate")
        self.assertEqual(tc_dict["function"]["arguments"], '{"expression": "100 / 4"}')

        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "c_123")
        self.assertEqual(messages[2]["content"], "25")


if __name__ == "__main__":
    unittest.main()

