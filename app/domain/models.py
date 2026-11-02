"""
Core project-thread-api/v1 domain models.

These models define the stable contract shared between
agents, HTTP endpoints, console applications, tests,
tracing, and monitoring.

Breaking changes require:

- contract version increment
- migration note
- caller updates
- contract test updates

Do not make silent schema edits.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal


Role = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]


GenerationFinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
]


AgentFinishReason = Literal[
    "final",
    "max_steps",
    "tool_error",
    "guardrail_blocked",
    "model_error",
]


@dataclass(frozen=True)
class Message:
    """
    Identity:
        Messages have no global identity.

    Ordering:
        Collections of messages MUST be ordered
        oldest -> newest.

    Tool messages:
        name and call_uid are populated only
        when role == 'tool'.
    """

    role: Role
    content: str
    name: str | None = None
    call_uid: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """
    Unknown is not zero.

    tokens_in:
        None means unavailable.

    tokens_out:
        None means unavailable.

    est_cost:
        None means the backend did not report one.
        Never invent a price.
    """

    text: str
    model_id: str

    tokens_in: int | None
    tokens_out: int | None

    est_cost: float | None

    finish_reason: GenerationFinishReason


class TextStream(Iterator[str]):
    """
    Stream of generated text chunks.

    Yields:
        str chunks for display as they arrive from the backend.

    Result guarantee:
        stream.result is the final GenerationResult for the same call,
        available ONLY after the stream is fully exhausted.
        Accessing result before exhaustion raises RuntimeError.
    """

    def __init__(
        self,
        chunk_iterator: Iterator[str],
        result_factory: Callable[[], GenerationResult],
    ) -> None:
        self._chunk_iterator = chunk_iterator
        self._result_factory = result_factory
        self._exhausted = False
        self._result: GenerationResult | None = None

    def __iter__(self) -> "TextStream":
        return self

    def __next__(self) -> str:
        try:
            return next(self._chunk_iterator)
        except StopIteration:
            if not self._exhausted:
                self._exhausted = True
                self._result = self._result_factory()
            raise

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted

    @property
    def result(self) -> GenerationResult:
        if not self._exhausted or self._result is None:
            raise RuntimeError(
                "Stream is not exhausted yet. Consume all chunks before accessing result."
            )
        return self._result


@dataclass(frozen=True)
class SearchResult:
    """
    Ordering:
        search results are best -> worst

    Score:
        cosine similarity

        higher = more relevant

    Metadata:
        may contain:
            page
            heading
            offsets

        must never contain:
            secrets
            api keys
            credentials
    """

    source_filename: str
    chunk_id: str | None
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class IndexManifest:
    """
    Manifest describing a vector index artifact.

    The manifest — not memory — records which embedding model,
    chunk settings, and corpus sources produced the index.
    """

    index_version: str
    embedding_model_id: str
    document_count: int
    chunk_count: int
    chunk_tokens: int
    overlap_tokens: int
    created_from: str


@dataclass(frozen=True)
class ToolDefinition:
    """
    Declaration of a tool executable by agents or callers.

    handler is code-side only and never serialized to the model.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]


@dataclass(frozen=True)
class ToolCall:
    """
    Tool invocation requested by the model.
    """

    call_uid: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolError:
    code: str
    message: str


@dataclass(frozen=True)
class ToolResult:
    """
    Boundary object returned from tools.

    Internal exceptions must be translated into
    ToolResult errors before crossing process,
    HTTP, UI, or MCP boundaries.
    """

    call_uid: str
    name: str
    ok: bool
    content: str
    data: dict[str, Any] | None = None
    error: ToolError | None = None


@dataclass(frozen=True)
class Usage:
    """
    Unknown values are None, not zero.
    """

    tokens_in: int | None
    tokens_out: int | None
    est_cost: float | None


@dataclass(frozen=True)
class AgentResult:
    """
    Top-level result returned by an agent.

    trace_id must match:

    - trace file
    - monitoring record
    - HTTP response

    for the same execution.
    """

    answer: str
    trace_id: str
    steps: int
    sources: list[SearchResult]
    tool_calls: list[ToolCall]
    usage: Usage
    finish_reason: AgentFinishReason
