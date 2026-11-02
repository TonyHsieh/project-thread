"""
Contract versioning & Migration Changelog.

Breaking changes require:
1. migration notes
2. contract test updates
3. caller updates
4. version increment

Never silently modify schemas.

Migration Changelog:
- v3.8: Breaking Change: Unified the correlation identifier for tool invocations
  under `call_uid` across ToolCall.call_uid (was ToolCall.call_id / id),
  ToolResult.call_uid (was ToolResult.call_id), and Message.call_uid (was Message.tool_call_id).
  Adheres to the "one value, one name, three fields" rule. Preserved unchanged even on failure.
  No aliases kept.
"""

CONTRACT_NAME = "project-thread-api"

CONTRACT_VERSION = "v1"

FULL_CONTRACT_ID = (
    f"{CONTRACT_NAME}/{CONTRACT_VERSION}"
)
