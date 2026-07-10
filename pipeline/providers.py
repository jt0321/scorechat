"""
pipeline/providers.py
Selects LangChain chat/embedding model backends at runtime via env vars,
so scorechat isn't locked to OpenAI:

    CHAT_PROVIDER=openai|anthropic|ollama   (default: openai)
    CHAT_MODEL=<model name>                 (provider-specific default if unset)
    EMBEDDING_PROVIDER=openai|ollama        (default: openai)
    EMBEDDING_MODEL=<model name>            (provider-specific default if unset)

Anthropic has no embeddings API, so EMBEDDING_PROVIDER=anthropic is rejected.

Note on embeddings: score_segments.embedding / text_sources.embedding are
declared `vector(1536)` in db/schema.sql. OpenAI's text-embedding-3-small
produces 1536-dim vectors. Switching EMBEDDING_PROVIDER to a model with a
different output dimension (e.g. Ollama's nomic-embed-text is 768-dim) will
fail on insert/query against the existing columns and requires migrating
the schema plus re-embedding all rows.
"""

from __future__ import annotations
import os

_CHAT_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-5",
    "ollama": "llama3.1",
}
_CHAT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

_EMBEDDING_DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
}
_EMBEDDING_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
}


def _is_placeholder(value: str) -> bool:
    return not value or "your-" in value or value.startswith("sk-placeholder")


def chat_provider() -> str:
    return os.environ.get("CHAT_PROVIDER", "openai").lower()


def embedding_provider() -> str:
    return os.environ.get("EMBEDDING_PROVIDER", "openai").lower()


def chat_provider_ready() -> bool:
    """Whether the configured chat provider has a usable (non-placeholder) key.
    Local providers (ollama) need no key and are always considered ready."""
    provider = chat_provider()
    key_env = _CHAT_KEY_ENV.get(provider)
    if key_env is None:
        return True
    return not _is_placeholder(os.environ.get(key_env, ""))


def embedding_provider_ready() -> bool:
    provider = embedding_provider()
    key_env = _EMBEDDING_KEY_ENV.get(provider)
    if key_env is None:
        return True
    return not _is_placeholder(os.environ.get(key_env, ""))


def get_chat_model(model: str | None = None, temperature: float = 0.3):
    """Returns a LangChain chat model for CHAT_PROVIDER (env-selected)."""
    provider = chat_provider()
    model = model or os.environ.get("CHAT_MODEL") or _CHAT_DEFAULT_MODELS.get(provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, api_key=os.environ["OPENAI_API_KEY"])

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                "CHAT_PROVIDER=anthropic requires langchain-anthropic. "
                "Install with: uv pip install langchain-anthropic"
            ) from e
        return ChatAnthropic(model=model, temperature=temperature, api_key=os.environ["ANTHROPIC_API_KEY"])

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "CHAT_PROVIDER=ollama requires langchain-ollama. "
                "Install with: uv pip install langchain-ollama"
            ) from e
        return ChatOllama(model=model, temperature=temperature, base_url=os.environ.get("OLLAMA_BASE_URL"))

    raise ValueError(f"Unknown CHAT_PROVIDER '{provider}'. Supported: openai, anthropic, ollama.")


def get_embeddings_model(model: str | None = None):
    """Returns a LangChain embeddings model for EMBEDDING_PROVIDER (env-selected)."""
    provider = embedding_provider()
    model = model or os.environ.get("EMBEDDING_MODEL") or _EMBEDDING_DEFAULT_MODELS.get(provider)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, api_key=os.environ["OPENAI_API_KEY"])

    if provider == "ollama":
        try:
            from langchain_ollama import OllamaEmbeddings
        except ImportError as e:
            raise ImportError(
                "EMBEDDING_PROVIDER=ollama requires langchain-ollama. "
                "Install with: uv pip install langchain-ollama"
            ) from e
        return OllamaEmbeddings(model=model, base_url=os.environ.get("OLLAMA_BASE_URL"))

    raise ValueError(
        f"Unknown or unsupported EMBEDDING_PROVIDER '{provider}'. Supported: openai, ollama "
        "(Anthropic has no embeddings API)."
    )
