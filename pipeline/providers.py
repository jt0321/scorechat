"""
pipeline/providers.py
Selects the LangChain chat backend at runtime via env vars,
so scorechat isn't locked to OpenAI:

    CHAT_PROVIDER=openai|anthropic|ollama|gemini|openrouter|cloudflare
                                                   (default: openai)
    CHAT_MODEL=<model name>                        (provider-specific default if unset)

`CHAT_PROVIDER` is the default, not the only choice: `chat_provider_options()`
reports every provider with the keys it needs and whether those keys are
present, so a caller -- `/api/providers`, and the picker in the web client --
can offer the ones that will actually work and pass the choice back through
`get_chat_model(provider=...)`. The keys stay here, server-side; only the
provider's name crosses to the browser.

Two of the providers are free tiers reached through OpenAI-compatible
endpoints, so they need no SDK of their own beyond langchain-openai:
OpenRouter (`:free` model slugs) and Cloudflare Workers AI (`/ai/v1`).
**Whichever provider is chosen has to support tool calling**, since that is how
ScoreChat answers at all; a model without it returns prose with an empty trace,
which is the one output this project treats as untrustworthy. The defaults
below are picked for that, and free catalogues churn -- OpenRouter's free slugs
come and go, so `CHAT_MODEL` is the escape hatch when a default disappears.

Embedding providers are listed here too, and embed only one thing: the prose
of the commentary texts. Score content is never embedded -- it is queried as
data. The two kinds of choice differ in scope. A chat model is chosen per
question; an embedding model is chosen per *corpus*, because a query has to be
embedded by the same model as the passages it is compared with. So embeddings
are stored per model (`passage_embeddings.model`, "provider:model"), several
can coexist, and a search can only use a model the corpus has been embedded
with. Every provider below has a free tier.
"""

from __future__ import annotations
import os

# The chat providers, in the order a picker should offer them. `requires` is
# every env var the provider needs before it can be called -- Cloudflare needs
# an account id as well as a key, because the account is part of the URL.
CHAT_PROVIDERS = {
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o",
        "requires": ["OPENAI_API_KEY"],
        "free_tier": False,
    },
    "anthropic": {
        "label": "Anthropic",
        "default_model": "claude-sonnet-5",
        "requires": ["ANTHROPIC_API_KEY"],
        "free_tier": False,
    },
    "gemini": {
        "label": "Google Gemini",
        "default_model": "gemini-2.5-flash",
        "requires": ["GEMINI_API_KEY"],
        "free_tier": True,
    },
    "openrouter": {
        "label": "OpenRouter (free tier)",
        "default_model": "nvidia/nemotron-3-super-120b-a12b:free",
        "requires": ["OPENROUTER_API_KEY"],
        "free_tier": True,
    },
    "cloudflare": {
        "label": "Cloudflare Workers AI (free tier)",
        "default_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "requires": ["CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID"],
        "free_tier": True,
    },
    "ollama": {
        "label": "Ollama (local)",
        "default_model": "llama3.1",
        "requires": [],
        "free_tier": True,
    },
}

_CHAT_DEFAULT_MODELS = {name: spec["default_model"] for name, spec in CHAT_PROVIDERS.items()}
_CHAT_KEY_ENV = {name: spec["requires"][0] for name, spec in CHAT_PROVIDERS.items() if spec["requires"]}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def _is_placeholder(value: str) -> bool:
    return not value or "your-" in value or value.startswith("sk-placeholder")


def chat_provider() -> str:
    return os.environ.get("CHAT_PROVIDER", "openai").lower()


def chat_provider_ready(provider: str | None = None) -> bool:
    """Whether a chat provider has usable (non-placeholder) credentials.

    Every var in `requires` must be present, not just the key: Cloudflare's
    account id is part of its URL, so a key without one cannot be called.
    Local providers (ollama) need nothing and are always ready.
    """
    provider = (provider or chat_provider()).lower()
    spec = CHAT_PROVIDERS.get(provider)
    if spec is None:
        return False
    return all(not _is_placeholder(os.environ.get(var, "")) for var in spec["requires"])


def chat_provider_options() -> list[dict]:
    """Every chat provider, with what it needs and whether it has it.

    What a picker is built from. `ready` is the only thing that decides whether
    an option can be chosen; the key values themselves never leave this process.
    """
    selected = chat_provider()
    return [
        {
            "id": name,
            "label": spec["label"],
            "default_model": (os.environ.get("CHAT_MODEL") if name == selected else None)
                             or spec["default_model"],
            "requires": list(spec["requires"]),
            "free_tier": spec["free_tier"],
            "ready": chat_provider_ready(name),
            "selected": name == selected,
        }
        for name, spec in CHAT_PROVIDERS.items()
    ]


def get_chat_model(model: str | None = None, temperature: float = 0.3,
                   provider: str | None = None):
    """Returns a LangChain chat model for `provider`, or for CHAT_PROVIDER.

    An explicit provider overrides the env default, and takes the provider's
    own default model with it: CHAT_MODEL belongs to CHAT_PROVIDER, and a
    Gemini model name passed to OpenRouter is a 404, not a fallback.
    """
    requested = (provider or "").lower() or None
    provider = requested or chat_provider()
    model = model or (os.environ.get("CHAT_MODEL") if requested is None else None) \
        or _CHAT_DEFAULT_MODELS.get(provider)

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

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "CHAT_PROVIDER=gemini requires langchain-google-genai. "
                "Install with: uv pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=os.environ["GEMINI_API_KEY"])

    # OpenAI-compatible endpoints: same client, different base URL. This is
    # what makes the free tiers cost nothing to support.
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature,
                          api_key=os.environ["OPENROUTER_API_KEY"],
                          base_url=OPENROUTER_BASE_URL)

    if provider == "cloudflare":
        from langchain_openai import ChatOpenAI

        # An assistant turn that only calls tools is sent with `content: null`,
        # which OpenAI accepts and Workers AI rejects ("'string' not in 'null'"),
        # so every answer failed on its second round trip, after the first tool
        # call. An empty string means the same thing to both.
        class WorkersAIChat(ChatOpenAI):
            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                for message in payload.get("messages", []):
                    if message.get("content") is None:
                        message["content"] = ""
                return payload

        account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
        return WorkersAIChat(model=model, temperature=temperature,
                             api_key=os.environ["CLOUDFLARE_API_KEY"],
                             base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1")

    raise ValueError(
        f"Unknown CHAT_PROVIDER '{provider}'. Supported: {', '.join(CHAT_PROVIDERS)}."
    )


# --- embeddings: commentary prose only ----------------------------------------------

# Defaults verified against each provider in September 2026. OpenAI-compatible
# endpoints get `check_embedding_ctx_length=False`: otherwise langchain-openai
# tokenises locally with tiktoken and sends token ids, which only OpenAI accepts.
EMBEDDING_PROVIDERS = {
    "gemini": {
        "label": "Google Gemini",
        "default_model": "gemini-embedding-2",
        "requires": ["GEMINI_API_KEY"],
    },
    "openrouter": {
        "label": "OpenRouter (free tier)",
        "default_model": "nvidia/nemotron-3-embed-1b:free",
        "requires": ["OPENROUTER_API_KEY"],
    },
    "cloudflare": {
        "label": "Cloudflare Workers AI (free tier)",
        "default_model": "@cf/baai/bge-base-en-v1.5",
        "requires": ["CLOUDFLARE_API_KEY", "CLOUDFLARE_ACCOUNT_ID"],
    },
    "ollama": {
        "label": "Ollama (local)",
        "default_model": "nomic-embed-text",
        "requires": [],
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "text-embedding-3-small",
        "requires": ["OPENAI_API_KEY"],
    },
}


def parse_embedding_model(spec: str) -> tuple[str, str]:
    """"gemini" or "gemini:gemini-embedding-2" -> ("gemini", "gemini-embedding-2")."""
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider not in EMBEDDING_PROVIDERS:
        raise ValueError(f"Unknown embedding provider '{provider}'. "
                         f"Supported: {', '.join(EMBEDDING_PROVIDERS)}.")
    return provider, model or EMBEDDING_PROVIDERS[provider]["default_model"]


def embedding_model_id(spec: str) -> str:
    """The name embeddings are stored under: always "provider:model", so the
    same model reached two ways ("gemini" and "gemini:gemini-embedding-2") is
    one set of vectors, not two."""
    provider, model = parse_embedding_model(spec)
    return f"{provider}:{model}"


def embedding_provider_ready(spec: str) -> bool:
    provider, _ = parse_embedding_model(spec)
    return all(not _is_placeholder(os.environ.get(var, ""))
               for var in EMBEDDING_PROVIDERS[provider]["requires"])


def get_embeddings(spec: str):
    """A LangChain embeddings model for "provider[:model]"."""
    provider, model = parse_embedding_model(spec)
    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model, google_api_key=os.environ["GEMINI_API_KEY"])
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, base_url=os.environ.get("OLLAMA_BASE_URL"))

    from langchain_openai import OpenAIEmbeddings
    if provider == "openai":
        return OpenAIEmbeddings(model=model, api_key=os.environ["OPENAI_API_KEY"])
    if provider == "openrouter":
        return OpenAIEmbeddings(model=model, api_key=os.environ["OPENROUTER_API_KEY"],
                                base_url=OPENROUTER_BASE_URL, check_embedding_ctx_length=False)
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    return OpenAIEmbeddings(model=model, api_key=os.environ["CLOUDFLARE_API_KEY"],
                            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1",
                            check_embedding_ctx_length=False)
