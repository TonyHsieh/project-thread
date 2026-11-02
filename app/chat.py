"""
Interactive chat adapter and conversational orchestration for project-thread-api/v1.

Built on top of stateless generation in textgen.py.
Provides:
- respond(history, system_prompt) -> GenerationResult
- respond_stream(history, system_prompt) -> TextStream
- chat_loop() -> runs interactive console adapter
"""

import sys
from typing import TextIO

from domain.errors import ModelError
from domain.models import GenerationResult, Message, TextStream
import textgen

DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful, and honest assistant."


def _validate_history(history: list[Message]) -> None:
    """
    Validate chat history contract:
    - Must be a non-empty list of Message instances.
    - Last message must have role == 'user'.
    """
    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("history must be a non-empty list of Message instances.")

    for idx, msg in enumerate(history):
        if not isinstance(msg, Message):
            raise TypeError(f"history element at index {idx} must be a domain Message instance.")

    last_message = history[-1]
    if last_message.role != "user":
        raise ValueError(
            f"The last message in history must have role 'user', got '{last_message.role}'."
        )


def respond(
    history: list[Message],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context: list[str] | None = None,
) -> GenerationResult:
    """
    Generate assistant response for the conversation history.

    Contract guarantees:
    - Requires the last message in history to have role 'user'.
    - Calls generate_text with that last message as prompt and earlier messages as history.
    - Forwards caller-provided context (ordered list of text passages) unchanged.
    - Does NOT mutate the caller's history list.
    """
    _validate_history(history)

    # Immutable slice: earlier messages only (oldest -> newest), excluding system prompt
    earlier_messages = list(history[:-1]) if len(history) > 1 else None
    prompt = history[-1].content

    return textgen.generate_text(
        prompt=prompt,
        system_prompt=system_prompt,
        history=earlier_messages,
        context=context,
    )


def respond_stream(
    history: list[Message],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    context: list[str] | None = None,
) -> TextStream:
    """
    Streaming variant of respond() for real-time display.

    Streaming changes display timing only; assembled text produces
    the identical GenerationResult contract.
    """
    _validate_history(history)

    earlier_messages = list(history[:-1]) if len(history) > 1 else None
    prompt = history[-1].content

    return textgen.generate_text_stream(
        prompt=prompt,
        system_prompt=system_prompt,
        history=earlier_messages,
        context=context,
    )


def chat_loop(
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    stream: bool = True,
    use_search: bool = False,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> None:
    """
    Run interactive console adapter loop.

    Rules:
    - Explicitly maintains in-memory history: appends user and assistant messages.
    - /reset: Clears in-memory history and confirms reset. Next response uses no prior facts.
    - /search or /rag: Toggles grounded retrieval preview on/off.
    - /exit, /quit: Exits chat session.
    - Streaming changes display timing only.
    - Grounding: Retrieval is a caller-side step; passes result.text values in rank order as context.
    """
    out = output_stream or sys.stdout
    history: list[Message] = []

    out.write("============================================================\n")
    out.write(" Project Thread Chat Console (type /reset to clear, /exit to quit)\n")
    if use_search:
        out.write(" [Grounded Retrieval Preview: ACTIVE (type /search to toggle)]\n")
    out.write("============================================================\n")
    out.flush()

    while True:
        try:
            if input_stream:
                line = input_stream.readline()
                if not line:
                    break
                user_input = line.strip()
            else:
                user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\nExiting session. Goodbye!\n")
            out.flush()
            break

        if not user_input:
            continue

        # Console adapter command: /reset
        if user_input.lower() == "/reset":
            history.clear()
            out.write("\n[System] Conversation history reset. Memory cleared.\n")
            out.flush()
            continue

        # Console adapter command: /search or /rag toggle
        if user_input.lower() in ("/search", "/rag"):
            use_search = not use_search
            status = "enabled" if use_search else "disabled"
            out.write(f"\n[System] Grounded retrieval {status}.\n")
            out.flush()
            continue

        # Console adapter command: /exit or /quit
        if user_input.lower() in ("/exit", "/quit", "/q"):
            out.write("\nExiting session. Goodbye!\n")
            out.flush()
            break

        # Caller-side retrieval step (optional preview, keeping generation decoupled)
        turn_context: list[str] | None = None
        if use_search:
            try:
                import search
                search_results = search.search_corpus(user_input, top_k=5)
                if search_results:
                    turn_context = [r.text for r in search_results]
                    out.write(f"\n[Grounding: retrieved {len(search_results)} document(s)]\n")
                    out.flush()
            except Exception as e:
                out.write(f"\n[Grounding warning: retrieval failed: {e}]\n")
                out.flush()

        # Append user message explicitly
        user_message = Message(role="user", content=user_input)
        history.append(user_message)

        try:
            if stream:
                out.write("\nAssistant: ")
                out.flush()
                text_stream = respond_stream(
                    history,
                    system_prompt=system_prompt,
                    context=turn_context,
                )
                for chunk in text_stream:
                    out.write(chunk)
                    out.flush()
                out.write("\n")
                out.flush()
                result = text_stream.result
            else:
                result = respond(
                    history,
                    system_prompt=system_prompt,
                    context=turn_context,
                )
                out.write(f"\nAssistant: {result.text}\n")
                out.flush()

            # Append assistant message explicitly
            history.append(Message(role="assistant", content=result.text))

        except ModelError as e:
            out.write(f"\n[Model Error] {e}\n")
            out.flush()
            # Rollback last user message so session state remains clean
            if history and history[-1] is user_message:
                history.pop()
        except Exception as e:
            out.write(f"\n[Unexpected Error] {e}\n")
            out.flush()
            if history and history[-1] is user_message:
                history.pop()


if __name__ == "__main__":
    enable_search = any(arg in sys.argv for arg in ("--search", "--rag"))
    chat_loop(use_search=enable_search)

