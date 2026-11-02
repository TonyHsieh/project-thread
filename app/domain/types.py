from typing import Literal

TraceId = str

ChunkId = str

ModelId = str

Role = Literal[
    "system",
    "user",
    "assistant",
    "tool",
]
