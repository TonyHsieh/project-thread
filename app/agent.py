"""
Autonomous Agent Loop for project-thread-api/v1 (Phase 13).

Orchestrates multi-step decision loops:
    Decide -> Emit ToolCall(s) -> call_tool() -> Observe -> Repeat
Terminates when the model returns a final text answer or reaches max_steps.
"""

from __future__ import annotations

import sys
from typing import Any
import uuid

from domain.errors import ModelError
from domain.models import AgentResult, Message, SearchResult, ToolCall, Usage
import textgen
import tools

DEFAULT_MAX_STEPS = 5


def run_agent(
    prompt: str,
    *,
    system_prompt: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    tools_definitions: list[dict[str, Any]] | None = None,
    model: str | None = None,
) -> AgentResult:
    """
    Run the autonomous agent loop for a user prompt.

    Contract guarantees:
    - prompt must be a non-empty string after trimming.
    - max_steps must be a positive integer.
    - trace_id is a unique identifier generated per execution.
    - steps counts the number of generation rounds (model decide calls).
    - sources contains SearchResult instances accumulated from search_corpus calls.
    - tool_calls contains all ToolCall instances requested across rounds in chronological order.
    - usage aggregates token counts and costs across all rounds; unknown fields are None.
    - finish_reason is 'final', 'max_steps', 'tool_error', or 'model_error'.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string after trimming.")

    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")

    trace_id = f"trace_{uuid.uuid4().hex[:12]}"
    available_tools = (
        tools_definitions
        if tools_definitions is not None
        else tools.get_openai_tools()
    )

    messages: list[Message] = [Message(role="user", content=prompt.strip())]
    accumulated_tool_calls: list[ToolCall] = []
    accumulated_sources: list[SearchResult] = []

    total_tokens_in: int = 0
    total_tokens_out: int = 0
    has_tokens_in: bool = False
    has_tokens_out: bool = False

    total_cost: float = 0.0
    has_cost: bool = False

    steps_count: int = 0
    last_assistant_text: str = ""

    while steps_count < max_steps:
        steps_count += 1

        try:
            result = textgen.generate_text(
                prompt=None,
                system_prompt=system_prompt,
                history=messages,
                tools=available_tools,
                model=model,
            )
        except ModelError as me:
            usage = Usage(
                tokens_in=total_tokens_in if has_tokens_in else None,
                tokens_out=total_tokens_out if has_tokens_out else None,
                est_cost=total_cost if has_cost else None,
            )
            return AgentResult(
                answer=f"Model error encountered: {me}",
                trace_id=trace_id,
                steps=steps_count,
                sources=accumulated_sources,
                tool_calls=accumulated_tool_calls,
                usage=usage,
                finish_reason="model_error",
            )

        # Accumulate usage adhering to The Null Rule
        if result.tokens_in is not None:
            total_tokens_in += result.tokens_in
            has_tokens_in = True
        if result.tokens_out is not None:
            total_tokens_out += result.tokens_out
            has_tokens_out = True
        if result.est_cost is not None:
            total_cost += result.est_cost
            has_cost = True

        if result.text:
            last_assistant_text = result.text

        # If model did not request tools or completed directly, finish
        if result.finish_reason != "tool_calls" or not result.tool_calls:
            usage = Usage(
                tokens_in=total_tokens_in if has_tokens_in else None,
                tokens_out=total_tokens_out if has_tokens_out else None,
                est_cost=total_cost if has_cost else None,
            )
            return AgentResult(
                answer=result.text,
                trace_id=trace_id,
                steps=steps_count,
                sources=accumulated_sources,
                tool_calls=accumulated_tool_calls,
                usage=usage,
                finish_reason="final",
            )

        # Append assistant message declaring the requested tool calls
        messages.append(
            Message(
                role="assistant",
                content=result.text,
                tool_calls=result.tool_calls,
            )
        )

        # Execute each requested tool call and append observation
        for tc in result.tool_calls:
            accumulated_tool_calls.append(tc)
            tool_res = tools.call_tool(tc.name, tc.arguments, call_uid=tc.call_uid)

            # Capture sources if search_corpus was called
            if tc.name == "search_corpus" and tool_res.ok and tool_res.data:
                for item in tool_res.data.get("results", []):
                    if isinstance(item, SearchResult):
                        accumulated_sources.append(item)
                    elif isinstance(item, dict):
                        accumulated_sources.append(
                            SearchResult(
                                source_filename=item.get("source_filename", ""),
                                chunk_id=item.get("chunk_id"),
                                text=item.get("text", ""),
                                score=item.get("score", 0.0),
                                metadata=item.get("metadata", {}),
                            )
                        )

            # Format tool observation
            if tool_res.ok:
                obs_content = tool_res.content
            else:
                err_code = tool_res.error.code if tool_res.error else "EXECUTION_ERROR"
                err_msg = tool_res.error.message if tool_res.error else tool_res.content
                obs_content = f"Tool Error ({err_code}): {err_msg}"

            messages.append(
                Message(
                    role="tool",
                    name=tc.name,
                    content=obs_content,
                    call_uid=tool_res.call_uid,
                )
            )

    # Exceeded max_steps without final answer
    usage = Usage(
        tokens_in=total_tokens_in if has_tokens_in else None,
        tokens_out=total_tokens_out if has_tokens_out else None,
        est_cost=total_cost if has_cost else None,
    )
    fallback_answer = (
        last_assistant_text
        if last_assistant_text
        else "Agent reached maximum step limit without completing."
    )
    return AgentResult(
        answer=fallback_answer,
        trace_id=trace_id,
        steps=steps_count,
        sources=accumulated_sources,
        tool_calls=accumulated_tool_calls,
        usage=usage,
        finish_reason="max_steps",
    )


def main() -> None:
    """CLI runner for running the autonomous agent on a prompt."""
    if len(sys.argv) < 2:
        print("Usage: python app/agent.py \"<prompt>\" [--steps <n>]")
        sys.exit(1)

    prompt = sys.argv[1]
    max_steps = DEFAULT_MAX_STEPS
    if "--steps" in sys.argv:
        idx = sys.argv.index("--steps")
        if idx + 1 < len(sys.argv):
            try:
                max_steps = int(sys.argv[idx + 1])
            except ValueError:
                pass

    print(f"🚀 Running agent on prompt: {prompt!r} (max_steps={max_steps})\n")
    result = run_agent(prompt, max_steps=max_steps)

    print(f"--- Agent Result [Trace: {result.trace_id}] ---")
    print(f"Finish Reason: {result.finish_reason}")
    print(f"Steps Taken:   {result.steps}")
    print(f"Tools Called:  {len(result.tool_calls)}")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"  [{i}] {tc.name}(arguments={tc.arguments}) (call_uid={tc.call_uid})")
    if result.sources:
        print(f"Sources Used:  {len(result.sources)}")
        for i, s in enumerate(result.sources, 1):
            print(f"  [{i}] {s.source_filename} (score: {s.score:.3f})")
    print(f"Tokens:        in={result.usage.tokens_in}, out={result.usage.tokens_out}")
    print("\nAnswer:")
    print(result.answer)


if __name__ == "__main__":
    main()

