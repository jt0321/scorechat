"""
pipeline/tools.py
The analysis API as tools a chat model can call, and the loop that runs them.

Why tool calling rather than retrieval. ScoreChat answers two shapes of
question -- "describe mm. x-y in the context of the whole work" and "find
recurring material" -- and neither is a retrieval problem. In the first the
measures are *given*, so there is nothing to search for; in the second the
answer is an edge in `span_relations`, which a vector search over prose
summaries cannot reach. Both are lookups, and a lookup wants a function call.

The rule the project rests on survives the change: the model chooses *which*
facts to fetch and how to phrase them, and invents none of them. Every tool
returns stored analysis, and a tool that finds nothing says so rather than
returning an empty result the model might fill in.
"""

from __future__ import annotations
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from pipeline import analysis_api
from pipeline.providers import chat_provider_ready, get_chat_model

MAX_TOOL_ITERATIONS = 8


def _text(content) -> str:
    """Flatten a reply to plain text.

    Providers differ: some return a string, others a list of content blocks
    (Gemini attaches a signature block to every reply). Rendering the list
    would put a base64 blob in front of the user.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [block.get("text", "") if isinstance(block, dict) else str(block)
                 for block in content]
        return "".join(part for part in parts if part)
    return str(content or "")


@tool
def resolve_work_tool(query: str) -> str:
    """Find which ingested movement a free-text description refers to.

    Call this first, before any other tool: everything else needs a work_id.
    Accepts nicknames, opus numbers and movement numbers in any usual form
    ("moonlight 3rd movement", "op 27 no 2 iii", "Waldstein mvt 1").
    If the result is ambiguous, ask the user which movement they mean rather
    than picking one.
    """
    return json.dumps(analysis_api.resolve_work(query))


@tool
def describe_span_tool(work_id: int, measure_start: int, measure_end: int) -> str:
    """Describe what happens in a range of bars: key regions, chords with their
    confidences, notated directions, and which engraved section it falls in.

    Bar numbers are printed numbering, as a performer reads them. Chords below
    the analyser's confidence threshold are absent rather than guessed, so a
    passage with few chords is an unlabelled passage -- say so.
    """
    return json.dumps(analysis_api.describe_span(work_id, measure_start, measure_end))


@tool
def find_recurrences_tool(work_id: int, measure_start: int, measure_end: int) -> str:
    """Find where else in the movement the material of these bars appears.

    Each result says which direction it runs: "returns_at" is a later passage
    restating the queried bars, "restates" means the queried bars are
    themselves the return of something earlier. `transposed_semitones` is 0
    when material comes back at pitch and 5 when it returns a fourth away;
    ignore it when `transposition_consistency` is low, since then no single
    interval explains the pair.
    """
    return json.dumps(analysis_api.find_recurrences(work_id, measure_start, measure_end))


@tool
def compare_spans_tool(
    work_id: int, a_start: int, a_end: int, b_start: int, b_end: int
) -> str:
    """Compare two specific bar ranges on demand.

    Use this when the user asks whether two particular passages are related and
    find_recurrences_tool returned nothing for them -- the stored graph keeps
    only matches above threshold, so a real question may have no stored answer.
    Report the confidences as measured, including low ones.
    """
    return json.dumps(analysis_api.compare_spans(work_id, a_start, a_end, b_start, b_end))


@tool
def get_key_plan_tool(work_id: int) -> str:
    """Get the movement's estimated key regions, as printed bar ranges.

    These are estimated, not notated, and the estimate is smoothed: a brief
    tonicisation is deliberately not reported as a region. When the whole
    movement comes back as one key the result says so — read that as the
    modulations being unconfirmed, not as the movement having none, and do not
    build a formal argument on the absence.
    """
    return json.dumps(analysis_api.get_key_plan(work_id))


@tool
def locate_in_form_tool(work_id: int, measure: int) -> str:
    """Find which engraved section a bar falls in, and the movement's repeat scheme.

    These sections come from the score's own expansion records, so they are
    notated rather than inferred -- but they carry letters, not form names. The
    score marks that a stretch repeats; it does not say it is an exposition.
    Draw the formal conclusion yourself from the scheme, the recurrences and
    the key plan, and say which of those support it.

    The scheme lists every play in order, and each section carries a
    play_count: any section played more than once is repeated. When a question
    is about repeats, report which sections repeat rather than only whether the
    first does — Op. 57's finale is notable precisely because its *second*
    section is the repeated one.
    """
    return json.dumps(analysis_api.locate_in_form(work_id, measure))


TOOLS = [
    resolve_work_tool, describe_span_tool, find_recurrences_tool,
    compare_spans_tool, get_key_plan_tool, locate_in_form_tool,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """You are ScoreChat, an expert musicologist writing for a musician.

You answer from stored score analysis, reached through the tools below. Never
state a musical fact the tools did not give you: no bar number you were not
told, no key you did not read, no formal label you did not derive from evidence
you can name. If the tools do not cover the question, say so plainly.

Work through a question in this order: resolve the movement first, then gather
what you need, then answer. Call several tools when a question needs them -- "is
this the recapitulation?" wants the recurrences, the repeat scheme and the key
plan, not one of the three.

Measure numbering is printed/engraved numbering, the numbering a performer
reads and the user types. Bar numbers belong to an *edition*: these are
transcribed from Durand 1915 (Paul Dukas, ed.), a performing edition, and
`reference_edition` on a resolved work says so. Where a repeat has first and
second endings, an urtext such as Henle numbers both endings from the same bar
(69 and 69b), while this edition numbers them straight through — so from the
second ending onward our numbers run ahead of an urtext's, by up to ten bars
in the movements that repeat most. When a bar number matters to the user and
the movement has alternate endings, say which edition the number is from. Bar 1 is the first complete measure. Some measures
are part of the score without being bars -- an anacrusis, the upbeat written
after a repeat barline, an unbarred cadenza -- and they have no number. When a
range is flagged as opening on a pickup, say "from the upbeat to m. N"; never
report a measure as bar 0.

On formal claims. The sections the tools return are engraved: the score marks
that a stretch of music repeats, not what to call it. Naming something an
exposition or a recapitulation is your inference, so make it one that rests on
stated evidence -- a repeat covering much of the movement, material returning at
pitch, a key plan that comes home -- and say which. Where the evidence is thin
or the analyser reports low confidence, prefer the weaker claim.

Write for a musician reading about the music, not a reader of a data dump. The
tools return every chord and direction in a range; select the ones that carry
the passage and leave the rest, and never paste the full list. Quote
confidences where they bear on a claim and round them sensibly. Cite bar ranges
as "mm. 17-20"."""


def _tool_result(call: dict) -> ToolMessage:
    tool_fn = TOOLS_BY_NAME.get(call["name"])
    if tool_fn is None:
        payload = json.dumps({"error": f"No such tool: {call['name']}"})
    else:
        try:
            payload = tool_fn.invoke(call["args"])
        except Exception as error:  # a bad argument is the model's to recover from
            payload = json.dumps({"error": f"{type(error).__name__}: {error}"})
    return ToolMessage(content=payload, tool_call_id=call["id"])


def answer(question: str, model: str | None = None) -> dict:
    """Answer one question, running whatever tool calls the model asks for.

    Returns the prose answer plus the full trace of tool calls, because the
    trace *is* the citation: it shows which stored analysis the answer rests
    on, and a claim with no supporting call in the trace is one to distrust.
    """
    if not chat_provider_ready():
        return {"answer": None, "trace": [],
                "error": "No chat provider is configured; set CHAT_PROVIDER and its API key."}

    llm = get_chat_model(model=model, temperature=0.2).bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]
    trace: list[dict] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        reply: AIMessage = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return {"answer": _text(reply.content), "trace": trace, "error": None}
        for call in reply.tool_calls:
            result = _tool_result(call)
            messages.append(result)
            trace.append({"tool": call["name"], "args": call["args"],
                          "result": json.loads(result.content)})

    # Out of iterations: answer from what was gathered rather than silently
    # truncating, and say that the search was cut short.
    messages.append(HumanMessage(
        content="Answer now from what you have gathered, and say that you "
                "stopped short of a full search."))
    reply = llm.invoke(messages)
    return {"answer": _text(reply.content), "trace": trace,
            "error": f"Stopped after {MAX_TOOL_ITERATIONS} rounds of tool calls."}
