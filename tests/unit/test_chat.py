"""
Unit and invariant tests for chat.py.

Covers:
- Contract: respond requires last message in history to have role 'user'
- Non-mutation: respond never mutates caller's history list
- History slicing: last message unpacked as prompt, earlier messages as history
- Streaming equivalence: respond_stream display timing vs GenerationResult parity
- Console adapter commands: /reset clears history, /exit terminates loop
"""

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

# Ensure app directory is in sys.path
_app_dir = Path(__file__).resolve().parent.parent.parent / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import chat
from domain.models import GenerationResult, Message, SearchResult, TextStream


class TestRespondContract(unittest.TestCase):
    """Test respond() contract invariants and validation."""

    def test_empty_history_raises_value_error(self):
        with self.assertRaises(ValueError):
            chat.respond([])

        with self.assertRaises(ValueError):
            chat.respond_stream([])

    def test_last_message_must_have_role_user(self):
        invalid_histories = [
            [Message(role="assistant", content="Hello")],
            [Message(role="tool", content="Result", call_uid="call-1")],
            [
                Message(role="user", content="Question 1"),
                Message(role="assistant", content="Answer 1"),
            ],
        ]
        for h in invalid_histories:
            with self.subTest(history=h):
                with self.assertRaises(ValueError):
                    chat.respond(h)
                with self.assertRaises(ValueError):
                    chat.respond_stream(h)

    def test_invalid_history_elements_raise_type_error(self):
        with self.assertRaises(TypeError):
            chat.respond(["string not Message"])  # type: ignore

    @patch("textgen.generate_text")
    def test_respond_does_not_mutate_caller_history(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            text="Assistant reply",
            model_id="test-model",
            tokens_in=5,
            tokens_out=2,
            est_cost=None,
            finish_reason="stop",
        )

        original_messages = [
            Message(role="user", content="First question"),
            Message(role="assistant", content="First answer"),
            Message(role="user", content="Follow up question"),
        ]
        # Make a shallow copy to check identity and length
        history_input = list(original_messages)
        original_length = len(history_input)

        res = chat.respond(history_input, system_prompt="You are helpful.")

        # Invariant: Caller's history list must NOT be mutated
        self.assertEqual(len(history_input), original_length)
        self.assertEqual(history_input, original_messages)
        self.assertEqual(res.text, "Assistant reply")

    @patch("textgen.generate_text")
    def test_history_slicing_single_and_multi_turn(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            text="Reply",
            model_id="test-model",
            tokens_in=5,
            tokens_out=2,
            est_cost=None,
            finish_reason="stop",
        )

        # Single turn
        single_turn = [Message(role="user", content="Single prompt")]
        chat.respond(single_turn, system_prompt="Custom prompt")
        mock_generate.assert_called_with(
            prompt="Single prompt",
            system_prompt="Custom prompt",
            history=None,
            context=None,
        )

        # Multi-turn
        multi_turn = [
            Message(role="user", content="Turn 1"),
            Message(role="assistant", content="Answer 1"),
            Message(role="user", content="Turn 2"),
        ]
        chat.respond(multi_turn, system_prompt="Custom prompt")
        mock_generate.assert_called_with(
            prompt="Turn 2",
            system_prompt="Custom prompt",
            history=[
                Message(role="user", content="Turn 1"),
                Message(role="assistant", content="Answer 1"),
            ],
            context=None,
        )

    @patch("textgen.generate_text")
    def test_respond_forwards_context_unchanged(self, mock_generate):
        mock_generate.return_value = GenerationResult(
            text="Reply with context",
            model_id="test-model",
            tokens_in=5,
            tokens_out=2,
            est_cost=None,
            finish_reason="stop",
        )
        history = [Message(role="user", content="Explain tokens")]
        context_docs = ["Passage 1", "Passage 2"]
        chat.respond(history, system_prompt="Sys", context=context_docs)

        mock_generate.assert_called_with(
            prompt="Explain tokens",
            system_prompt="Sys",
            history=None,
            context=context_docs,
        )


class TestRespondStream(unittest.TestCase):
    """Test respond_stream() streaming adapter."""

    @patch("textgen.generate_text_stream")
    def test_respond_stream_delegation_and_equivalence(self, mock_stream):
        expected_result = GenerationResult(
            text="Streamed reply",
            model_id="test-model",
            tokens_in=8,
            tokens_out=2,
            est_cost=None,
            finish_reason="stop",
        )
        dummy_stream = TextStream(
            chunk_iterator=iter(["Streamed ", "reply"]),
            result_factory=lambda: expected_result,
        )
        mock_stream.return_value = dummy_stream

        history = [Message(role="user", content="Hello stream")]
        stream = chat.respond_stream(history, system_prompt="Sys")

        mock_stream.assert_called_once_with(
            prompt="Hello stream",
            system_prompt="Sys",
            history=None,
            context=None,
        )

        chunks = list(stream)
        self.assertEqual(chunks, ["Streamed ", "reply"])
        self.assertEqual(stream.result, expected_result)

    @patch("textgen.generate_text_stream")
    def test_respond_stream_forwards_context_unchanged(self, mock_stream):
        mock_stream.return_value = TextStream(
            chunk_iterator=iter(["chunk"]),
            result_factory=lambda: GenerationResult(
                text="chunk",
                model_id="test-model",
                tokens_in=1,
                tokens_out=1,
                est_cost=None,
                finish_reason="stop",
            ),
        )
        history = [Message(role="user", content="Stream with context")]
        context_docs = ["Evidence A"]
        chat.respond_stream(history, system_prompt="Sys", context=context_docs)

        mock_stream.assert_called_once_with(
            prompt="Stream with context",
            system_prompt="Sys",
            history=None,
            context=context_docs,
        )


class TestChatLoop(unittest.TestCase):
    """Test interactive console adapter loop and commands."""

    @patch("chat.respond_stream")
    def test_chat_loop_reset_command_clears_history(self, mock_respond_stream):
        # Setup mock stream behavior for turn 1 and turn 2
        def make_stream(text):
            return TextStream(
                chunk_iterator=iter([text]),
                result_factory=lambda: GenerationResult(
                    text=text,
                    model_id="test-model",
                    tokens_in=5,
                    tokens_out=2,
                    est_cost=None,
                    finish_reason="stop",
                ),
            )

        call_histories = []

        def capture_call(history, system_prompt=None, context=None, **kwargs):
            call_histories.append(list(history))
            return make_stream(f"Answer to {history[-1].content}")

        mock_respond_stream.side_effect = capture_call

        # Simulated user session:
        # Turn 1: "Hello first"
        # Reset: "/reset"
        # Turn 2: "Hello after reset"
        # Exit: "/exit"
        session_input = io.StringIO("Hello first\n/reset\nHello after reset\n/exit\n")
        session_output = io.StringIO()

        chat.chat_loop(
            system_prompt="Test sys",
            stream=True,
            input_stream=session_input,
            output_stream=session_output,
        )

        output_text = session_output.getvalue()
        self.assertIn("Conversation history reset. Memory cleared.", output_text)
        self.assertIn("Exiting session. Goodbye!", output_text)

        # Invariant check:
        # First call has 1 message: ["Hello first"]
        self.assertEqual(len(call_histories[0]), 1)
        self.assertEqual(call_histories[0][0].content, "Hello first")

        # After /reset, second call MUST NOT contain prior turn facts!
        self.assertEqual(len(call_histories[1]), 1)
        self.assertEqual(call_histories[1][0].content, "Hello after reset")

    @patch("chat.respond")
    def test_chat_loop_non_streaming_mode(self, mock_respond):
        mock_respond.return_value = GenerationResult(
            text="Non-stream response",
            model_id="test-model",
            tokens_in=5,
            tokens_out=3,
            est_cost=None,
            finish_reason="stop",
        )

        session_input = io.StringIO("Query\n/quit\n")
        session_output = io.StringIO()

        chat.chat_loop(
            system_prompt="Test sys",
            stream=False,
            input_stream=session_input,
            output_stream=session_output,
        )

        output_text = session_output.getvalue()
        self.assertIn("Assistant: Non-stream response", output_text)
        self.assertIn("Exiting session. Goodbye!", output_text)

    @patch("chat.respond_stream")
    @patch("search.search_corpus")
    def test_chat_loop_search_grounding(self, mock_search, mock_respond_stream):
        mock_search.return_value = [
            SearchResult(
                source_filename="doc.txt",
                chunk_id="doc.txt:0",
                text="Retrieved document text",
                score=0.9,
                metadata={},
            )
        ]
        mock_respond_stream.return_value = TextStream(
            chunk_iterator=iter(["Grounded answer"]),
            result_factory=lambda: GenerationResult(
                text="Grounded answer",
                model_id="test-model",
                tokens_in=5,
                tokens_out=2,
                est_cost=None,
                finish_reason="stop",
            ),
        )

        session_input = io.StringIO("What is attention?\n/exit\n")
        session_output = io.StringIO()

        chat.chat_loop(
            system_prompt="Test sys",
            stream=True,
            use_search=True,
            input_stream=session_input,
            output_stream=session_output,
        )

        output_text = session_output.getvalue()
        self.assertIn("retrieved 1 document(s)", output_text)
        mock_search.assert_called_once_with("What is attention?", top_k=5)
        # Check that context=[r.text] was passed to respond_stream
        call_kwargs = mock_respond_stream.call_args.kwargs
        self.assertEqual(call_kwargs.get("context"), ["Retrieved document text"])

    def test_chat_loop_toggle_search_command(self):
        session_input = io.StringIO("/search\n/exit\n")
        session_output = io.StringIO()

        chat.chat_loop(
            system_prompt="Test sys",
            use_search=False,
            input_stream=session_input,
            output_stream=session_output,
        )

        output_text = session_output.getvalue()
        self.assertIn("Grounded retrieval enabled.", output_text)


if __name__ == "__main__":
    unittest.main()

