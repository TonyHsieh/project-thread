class AgentError(Exception):
    """Base agent exception."""

class ModelError(AgentError):
    pass

class ToolExecutionError(AgentError):
    pass

class SearchError(AgentError):
    pass

class GuardrailError(AgentError):
    pass
