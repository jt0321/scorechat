"""
tests/test_providers.py
Which chat backends this deployment can offer, and what picking one does.

No network and no keys: a LangChain chat model is constructed, not called, so
what is under test is the wiring -- that a provider's credentials decide
whether it is offered, that choosing one overrides the env default without
dragging CHAT_MODEL along with it, and that the OpenAI-compatible free tiers
point at the right base URL. Getting that last one wrong sends an OpenRouter
key to api.openai.com.
"""

import pytest

from pipeline import providers


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("CHAT_PROVIDER", "CHAT_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY", "OPENROUTER_API_KEY", "CLOUDFLARE_API_KEY",
                "CLOUDFLARE_ACCOUNT_ID"):
        monkeypatch.delenv(var, raising=False)


def test_a_provider_without_its_key_is_offered_but_not_ready(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    options = {o["id"]: o for o in providers.chat_provider_options()}

    assert options["openrouter"]["ready"] is True
    # Offered rather than hidden: a picker that silently drops a provider gives
    # the user no way to learn that it only wants a key.
    assert options["gemini"]["ready"] is False
    assert options["gemini"]["requires"] == ["GEMINI_API_KEY"]


def test_cloudflare_needs_the_account_id_as_well_as_the_key(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-token")
    assert providers.chat_provider_ready("cloudflare") is False

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
    assert providers.chat_provider_ready("cloudflare") is True


def test_free_tiers_are_openai_compatible_endpoints(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")

    router = providers.get_chat_model(provider="openrouter")
    assert str(router.openai_api_base) == providers.OPENROUTER_BASE_URL
    assert router.model_name.endswith(":free")

    workers = providers.get_chat_model(provider="cloudflare")
    assert "accounts/acct-123/ai/v1" in str(workers.openai_api_base)


def test_cloudflare_is_never_sent_null_content(monkeypatch):
    """An assistant turn that only calls tools goes out as `content: null`,
    which Workers AI rejects -- so every answer failed after its first tool
    call. The replay of that turn must carry a string."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
    workers = providers.get_chat_model(provider="cloudflare")

    call = {"name": "resolve_work_tool", "args": {"query": "Moonlight"}, "id": "call_1"}
    payload = workers._get_request_payload([
        HumanMessage("where is the recapitulation?"),
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content="{}", tool_call_id="call_1"),
    ])
    assert all(isinstance(m.get("content"), str) for m in payload["messages"])


def test_picking_a_provider_does_not_carry_the_env_model_over(monkeypatch):
    """CHAT_MODEL belongs to CHAT_PROVIDER. A Gemini model name passed to
    OpenRouter is a 404, not a fallback, so an explicit provider takes its own
    default instead."""
    monkeypatch.setenv("CHAT_PROVIDER", "gemini")
    monkeypatch.setenv("CHAT_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")

    model = providers.get_chat_model(provider="openrouter")
    assert model.model_name == providers.CHAT_PROVIDERS["openrouter"]["default_model"]

    # An explicit model still wins over the provider's default.
    named = providers.get_chat_model(provider="openrouter", model="some/other:free")
    assert named.model_name == "some/other:free"


def test_an_unknown_provider_is_refused_rather_than_defaulted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert providers.chat_provider_ready("not-a-provider") is False
    with pytest.raises(ValueError, match="Unknown CHAT_PROVIDER"):
        providers.get_chat_model(provider="not-a-provider")
