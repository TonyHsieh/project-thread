"""
project-thread-api/v1

Global contract invariants.

These are cross-cutting rules that every caller
must honor.

Ordering:

    history messages are oldest -> newest

    context preserves supplied order

Identity:

    trace_id is generated per request

Unknown Is Not Zero:

    tokens_in=None
    tokens_out=None
    est_cost=None

means unknown.

A reported value of 0 is a real measurement.

Compatibility:

    Breaking changes require:

    - migration note
    - contract test updates
    - caller updates
    - contract version change
"""

CONTRACT_VERSION = "project-thread-api/v1"

ORDERING_RULES = {
    "messages": "oldest_to_newest",
    "search_results": "best_to_worst",
    "steps": "start_at_1_increment_by_1",
}

IDENTITY_RULES = {
    "trace_id_required": True,
    "trace_id_global": True,
}

UNKNOWN_VALUE_RULES = {
    "tokens": "null_not_zero",
    "costs": "null_not_zero",
}

TIMESTAMP_RULE = "ISO8601 UTC"
