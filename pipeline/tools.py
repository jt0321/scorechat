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

from db.store import list_works
from pipeline import analysis_api
from pipeline.providers import chat_provider_ready, get_chat_model

MAX_TOOL_ITERATIONS = 8

# How much of a conversation a question carries. Enough for a follow-up ("and
# the second movement?") or a reply to a clarifying question to make sense;
# small enough that a free-tier context window and rate budget survive it.
# Earlier turns travel as their prose only -- never their tool results, which
# are most of a turn's tokens -- plus the works they touched, so a follow-up
# can reuse a work_id instead of resolving the work again.
MAX_HISTORY_TURNS = 6
MAX_HISTORY_QUESTION_CHARS = 1_000
MAX_HISTORY_ANSWER_CHARS = 1_500
MAX_CONTEXT_WORKS = 12


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
    """Find which ingested movement a free-text description refers to, and the
    sonata it belongs to.

    Call this first, before any other tool: everything else needs a work_id.
    Pass the user's own wording. Accepts nicknames, opus numbers however they
    are spaced or punctuated ("op31/no3", "Op. 31, No. 3"), movement numbers
    ("moonlight 3rd movement", "op 27 no 2 iii") and movement headings ("the
    Scherzo of the Hunt", "the fugue of Op. 106").
    `sonata` lists every movement of the sonata with its heading and engraved
    key: use it when the question is about the whole work. If no single
    movement is resolved and the question needs one, ask which is meant rather
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


@tool
def search_commentary_tool(query: str, work_id: int | None = None, whole_sonata: bool = False) -> str:
    """Search what published commentators wrote about the sonatas: Elterlein
    (1879, all 32 sonatas), Marx (1895, twenty of them), Shedlock (1895, a
    history of the sonata).

    Use it for questions about character, interpretation, history, or what
    writers have said -- never as evidence of a musical fact, which comes only
    from the score tools. Pass `work_id` to keep to one movement (remarks on
    its sonata as a whole come back too, marked scope "sonata"), and
    `whole_sonata` to widen to every movement. Quote briefly, name the author
    and page, and say it is their view. Bar numbers in these texts follow the
    author's edition, not ours: never cite them as ours.
    """
    return json.dumps(analysis_api.search_commentary(query, work_id, whole_sonata))


@tool
def commentary_claims_tool(work_id: int) -> str:
    """What the commentators assert about a movement -- keys, formal terms, bar
    references -- and, for keys, what the score says about each assertion.

    `supported` means the movement's engraved key bears the claim out.
    `agrees_with_estimate` / `disagrees_with_estimate` compare it with the
    *estimated* key regions, which smooth away brief modulations: when a
    commentator names a key the estimate lacks, say the two disagree and do
    not assume the commentator is wrong. Formal terms are interpretations.
    Use this when the user asks whether the commentary is right, or how the
    writers' account compares with the score.
    """
    return json.dumps(analysis_api.commentary_claims(work_id))


TOOLS = [
    resolve_work_tool, describe_span_tool, find_recurrences_tool,
    compare_spans_tool, get_key_plan_tool, locate_in_form_tool,
    search_commentary_tool, commentary_claims_tool,
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


SYSTEM_PROMPT = """You are ScoreChat, an expert musicologist writing for a musician.

You answer from stored score analysis, reached through the tools below. Never
state a musical fact the tools did not give you: no bar number you were not
told, no key you did not read, no formal label you did not derive from evidence
you can name. If the tools do not cover the question, say so plainly.

The *work* is the whole sonata: Op. 31 No. 3 ("The Hunt") is one work in four
movements, the third of three sonatas published together as Op. 31. Each
movement has its own work_id, the way a recording gives each movement its own
track, and musicians often name a movement by its heading rather than its
number ("the Scherzo", "the fugue" for Op. 106/iv). When a question names a
sonata without a movement, `sonata` in resolve_work_tool's result gives every
movement's heading and key.

A question may follow earlier ones in the same conversation. Read it against
them: "the second movement", "that passage", "and in the recapitulation?" mean
the work and bars already under discussion, and a short reply to a question you
asked ("the scherzo", "Op. 31 No. 3") answers that question -- combine it with
the original request rather than treating it as a new one. When a question is
genuinely ambiguous, ask one short clarifying question instead of guessing.
Earlier answers are your own prose, not tool results: re-fetch any fact you
need to state again rather than repeating it from memory.

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

Two kinds of source reach you and they must never be confused. The score tools
return what the score shows. The commentary tools return what nineteenth-century
writers -- Elterlein, Marx, Shedlock -- *said* about it: attribute every such
remark to its author ("Marx hears the finale as ..."), quote briefly with the
page, and never present it as fact. Where a commentator's claim and the score
disagree, say so plainly and let the score decide matters of fact; where the
disagreement is with an *estimated* key, say that the estimate may be what is
wrong. A bar number in the commentary is the author's edition's: never repeat
it as one of ours. Reach for the commentary when a question is about character,
meaning, history or interpretation, or asks what writers have said -- not to
fill a gap the score tools left.

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


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def clean_history(history) -> list[dict]:
    """The last few turns of a conversation, in a shape safe to send a model.

    History arrives from the client, so it is validated here rather than
    trusted: wrong types are dropped, text is clipped, and only the most recent
    turns are kept, so a long conversation cannot inflate every request.
    """
    turns = []
    for turn in history if isinstance(history, list) else []:
        if not isinstance(turn, dict):
            continue
        question, reply = turn.get("question"), turn.get("answer")
        if not isinstance(question, str) or not question.strip():
            continue
        work_ids = [w for w in turn.get("work_ids") or []
                    if isinstance(w, int) and not isinstance(w, bool)]
        turns.append({
            "question": _clip(question.strip(), MAX_HISTORY_QUESTION_CHARS),
            "answer": _clip(reply.strip(), MAX_HISTORY_ANSWER_CHARS)
                      if isinstance(reply, str) and reply.strip() else None,
            "work_ids": work_ids,
        })
    return turns[-MAX_HISTORY_TURNS:]


def context_work_ids(trace: list[dict]) -> list[int]:
    """The movements a turn was about, most recent last: what it resolved, what
    its other calls were made on, and -- when it named a whole sonata -- that
    sonata's movements, so "the second one" can be answered next turn."""
    ids: list[int] = []
    for step in trace:
        result, args = step.get("result") or {}, step.get("args") or {}
        if step.get("tool") == "resolve_work_tool":
            if result.get("resolved"):
                ids.append(result["resolved"]["work_id"])
            elif result.get("sonata"):
                ids.extend(m["work_id"] for m in result["sonata"]["movements"])
        if isinstance(args.get("work_id"), int):
            ids.append(args["work_id"])
    return list(dict.fromkeys(ids))


def _works_in_context(history: list[dict]) -> str | None:
    """A short list of the works earlier turns were about, for the prompt.

    The ids come from the client, so each is looked up again here and one the
    corpus does not have is simply dropped; what the model reads is always the
    catalogue's own description of the work.
    """
    wanted = list(dict.fromkeys(w for turn in reversed(history) for w in turn["work_ids"]))
    catalogue = {w["id"]: w for w in list_works()} if wanted else {}
    lines = []
    for work_id in wanted[:MAX_CONTEXT_WORKS]:
        work = catalogue.get(work_id)
        if work is None:
            continue
        nickname = f" ({work['nickname']})" if work.get("nickname") else ""
        heading = f", {work['tempo_indication']}" if work.get("tempo_indication") else ""
        lines.append(f"- work_id {work_id}: {work.get('opus')}{nickname}, "
                     f"movement {work.get('movement_number')}{heading}")
    return "\n".join(lines) or None


def _conversation(question: str, history: list[dict]) -> list:
    system = SYSTEM_PROMPT
    works = _works_in_context(history)
    if works:
        system += ("\n\nWorks discussed earlier in this conversation, most recent "
                   "first. A follow-up about one of these can use its work_id "
                   "directly, without resolving it again:\n" + works)
    messages = [SystemMessage(content=system)]
    for turn in history:
        messages.append(HumanMessage(content=turn["question"]))
        # A turn that failed still happened: the user asked it, and the next
        # question may be a rephrasing of it.
        messages.append(AIMessage(content=turn["answer"] or "(No answer was given.)"))
    messages.append(HumanMessage(content=question))
    return messages


def answer(question: str, model: str | None = None, provider: str | None = None,
           history: list[dict] | None = None) -> dict:
    """Answer one question, running whatever tool calls the model asks for.

    Returns the prose answer plus the full trace of tool calls, because the
    trace *is* the citation: it shows which stored analysis the answer rests
    on, and a claim with no supporting call in the trace is one to distrust.

    `provider` picks a backend for this question alone, leaving CHAT_PROVIDER
    as the default; the caller passes a name, never a key. A provider whose
    model cannot call tools will answer with an empty trace, which is the
    signal to distrust the answer rather than a failure to report here.

    `history` is the conversation so far, as `{question, answer, work_ids}`
    turns (see `clean_history`); the result's `context_work_ids` is this turn's
    entry for the next one.
    """
    if not chat_provider_ready(provider):
        which = provider or "the configured provider"
        return {"answer": None, "trace": [], "context_work_ids": [],
                "error": f"No API key is configured for {which}; see .env.example."}

    llm = get_chat_model(model=model, temperature=0.2, provider=provider).bind_tools(TOOLS)
    messages = _conversation(question, clean_history(history))
    trace: list[dict] = []

    for _ in range(MAX_TOOL_ITERATIONS):
        reply: AIMessage = llm.invoke(messages)
        messages.append(reply)
        if not reply.tool_calls:
            return {"answer": _text(reply.content), "trace": trace,
                    "context_work_ids": context_work_ids(trace), "error": None}
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
            "context_work_ids": context_work_ids(trace),
            "error": f"Stopped after {MAX_TOOL_ITERATIONS} rounds of tool calls."}
