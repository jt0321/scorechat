"""
tests/test_tools.py
The tool-calling loop, driven by a stub model.

No API key and no network: the point is the loop's behaviour, not the model's
judgement. What must hold is that tool results reach the model, that a bad call
comes back as an error the model can recover from rather than an exception, and
that a model which never stops calling tools is cut off with an answer rather
than looping.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from pipeline import tools


class StubModel:
    """Replays a scripted sequence of replies and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen: list[list] = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0) if self.replies else AIMessage(content="done")


def call(name, args, id="1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


@pytest.fixture
def stub(monkeypatch):
    def install(replies):
        model = StubModel(replies)
        monkeypatch.setattr(tools, "get_chat_model", lambda **kwargs: model)
        monkeypatch.setattr(tools, "chat_provider_ready", lambda *a, **k: True)
        return model
    return install


def test_a_tool_result_is_sent_back_to_the_model(stub, monkeypatch):
    monkeypatch.setattr(tools.analysis_api, "get_key_plan",
                        lambda *a, **k: {"regions": [{"value": "f minor"}]})
    model = stub([
        AIMessage(content="", tool_calls=[call("get_key_plan_tool", {"work_id": 1})]),
        AIMessage(content="It is in F minor."),
    ])
    result = tools.answer("what key?")
    assert result["answer"] == "It is in F minor."
    assert result["trace"][0]["result"] == {"regions": [{"value": "f minor"}]}
    assert "f minor" in str(model.seen[-1])


def test_an_unknown_tool_comes_back_as_an_error_not_an_exception(stub):
    stub([
        AIMessage(content="", tool_calls=[call("no_such_tool", {})]),
        AIMessage(content="I could not do that."),
    ])
    result = tools.answer("anything")
    assert "No such tool" in result["trace"][0]["result"]["error"]
    assert result["answer"] == "I could not do that."


def test_a_failing_tool_is_reported_to_the_model_to_recover_from(stub, monkeypatch):
    def explode(*args, **kwargs):
        raise ValueError("measure 9999 is past the end")
    monkeypatch.setattr(tools.analysis_api, "describe_span", explode)
    stub([
        AIMessage(content="", tool_calls=[
            call("describe_span_tool", {"work_id": 1, "measure_start": 9999, "measure_end": 9999})]),
        AIMessage(content="That bar does not exist."),
    ])
    result = tools.answer("describe m. 9999")
    assert "ValueError" in result["trace"][0]["result"]["error"]
    assert result["answer"] == "That bar does not exist."


def test_several_calls_in_one_reply_all_run(stub, monkeypatch):
    monkeypatch.setattr(tools.analysis_api, "get_key_plan", lambda *a, **k: {"regions": []})
    monkeypatch.setattr(tools.analysis_api, "locate_in_form",
                        lambda *a, **k: {"section": None})
    stub([
        AIMessage(content="", tool_calls=[
            call("get_key_plan_tool", {"work_id": 1}, "a"),
            call("locate_in_form_tool", {"work_id": 1, "measure": 1}, "b"),
        ]),
        AIMessage(content="Both answered."),
    ])
    result = tools.answer("two things")
    assert [step["tool"] for step in result["trace"]] == [
        "get_key_plan_tool", "locate_in_form_tool"]


def test_an_endless_caller_is_cut_off_with_an_answer(stub, monkeypatch):
    """A model that keeps calling tools must not loop; it is asked to answer
    from what it has and the answer says the search was cut short."""
    monkeypatch.setattr(tools.analysis_api, "get_key_plan", lambda *a, **k: {"regions": []})
    forever = [AIMessage(content="", tool_calls=[call("get_key_plan_tool", {"work_id": 1})])
               for _ in range(tools.MAX_TOOL_ITERATIONS)]
    stub(forever + [AIMessage(content="Partial answer.")])
    result = tools.answer("loop please")
    assert result["answer"] == "Partial answer."
    assert "Stopped after" in result["error"]
    assert len(result["trace"]) == tools.MAX_TOOL_ITERATIONS


def test_no_provider_is_an_error_rather_than_a_fabricated_answer(monkeypatch):
    monkeypatch.setattr(tools, "chat_provider_ready", lambda *a, **k: False)
    result = tools.answer("anything")
    assert result["answer"] is None and "No API key" in result["error"]


def test_a_picked_provider_is_named_in_the_error_and_passed_to_the_model(monkeypatch):
    """Picking a provider whose key is missing must say which one: the whole
    point of the picker is that the user chose it, so falling back silently to
    the env default would answer from a model they did not choose."""
    monkeypatch.setattr(tools, "chat_provider_ready", lambda provider=None: provider != "cloudflare")
    assert "cloudflare" in tools.answer("anything", provider="cloudflare")["error"]

    seen = {}
    monkeypatch.setattr(tools, "get_chat_model",
                        lambda **kwargs: seen.update(kwargs) or StubModel([AIMessage(content="ok")]))
    tools.answer("anything", provider="openrouter", model="some/model:free")
    assert seen["provider"] == "openrouter" and seen["model"] == "some/model:free"


def test_content_blocks_are_flattened_to_text():
    """Gemini attaches a signature block to every reply; rendering the list
    would put a base64 blob in front of the user."""
    blocks = [{"type": "text", "text": "The recapitulation is at m. 103."},
              {"type": "signature", "extras": {"signature": "AAAA"}}]
    assert tools._text(blocks) == "The recapitulation is at m. 103."
    assert tools._text("plain") == "plain"


def test_every_tool_documents_itself_for_the_model():
    """The description is the only instruction the model gets about when to
    call a tool, so an undocumented tool is an uncallable one."""
    for tool_fn in tools.TOOLS:
        assert tool_fn.description and len(tool_fn.description) > 80, tool_fn.name
