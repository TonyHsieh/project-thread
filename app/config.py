"""
Configuration management for model inference and service providers.

Loads environment variables and provides client and default model settings
without leaking provider specifics into generation callers.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Automatically load environment variables from local or parent directory
_current_dir = Path(__file__).resolve().parent
_env_candidates = [
    _current_dir / ".env",
    _current_dir.parent / ".env",
]
for _env_file in _env_candidates:
    if _env_file.is_file():
        load_dotenv(dotenv_path=_env_file, override=False)


def get_required_env(var_name: str) -> str:
    """Get a required environment variable or raise an error with a clear message."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {var_name}. Please set it in your .env file."
        )
    return value


def get_provider() -> str:
    """
    Determine active provider: 'azure' or 'openai'.
    Can be explicitly configured via AI_PROVIDER or auto-detected.
    """
    explicit = os.getenv("AI_PROVIDER")
    if explicit:
        return explicit.lower()
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return "azure"
    return "openai"


def get_default_model_id() -> str:
    """Return the default model id configured for the active environment."""
    explicit_default = os.getenv("DEFAULT_MODEL_ID")
    if explicit_default:
        return explicit_default

    provider = get_provider()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    return os.getenv("OPENAI_MODEL_ID", "gpt-4o")


def get_default_embedding_model_id() -> str:
    """
    Return the default embedding model id configured for the active environment.
    Guarantees Comparability rule: shared across Phase 1, Phase 5, and Agentic retrieval.
    """
    explicit = os.getenv("DEFAULT_EMBEDDING_MODEL_ID")
    if explicit:
        return explicit

    provider = get_provider()
    if provider == "azure":
        return os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    return os.getenv("OPENAI_EMBEDDING_MODEL_ID", "text-embedding-3-small")


def get_client() -> OpenAI:
    """
    Construct and return configured OpenAI client for the active provider.
    Ensures credentials remain isolated behind this boundary.
    """
    provider = get_provider()
    match provider:
        case "azure":
            api_key = get_required_env("AZURE_OPENAI_API_KEY")
            endpoint = get_required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
            return OpenAI(
                api_key=api_key,
                base_url=f"{endpoint}/openai/v1/",
            )
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required when using the 'openai' provider.")
            return OpenAI(api_key=api_key)
        case _:
            raise ValueError(f"Unknown provider mode: {provider}")

