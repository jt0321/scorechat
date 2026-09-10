"""
pipeline/chat.py
Scorechat RAG: retrieve score segments + text sources, then generate
a grounded LLM response with measure-level citations via a LangChain
LCEL chain (prompt | chat model | StrOutputParser). The chat model
backend is selected by CHAT_PROVIDER (see pipeline/providers.py) so
this isn't locked to OpenAI.
"""

from __future__ import annotations
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipeline.providers import get_chat_model, chat_provider_ready
from pipeline.retrieval import retrieve
from db.store import get_measure_evidence

MAX_SYMBOLIC_CONTEXT_MEASURES = 24

SYSTEM_PROMPT = """You are Scorechat, an expert musicologist and piano pedagogue.
Answer questions about musical scores using the retrieved score excerpts and
reference materials provided below. Cite specific measure ranges when discussing
musical passages (e.g. "mm. 17–20"). Be precise about harmony, texture, and form.

Measure numbering follows normal engraving convention, the same numbering a
performer reads off the printed page: bar 1 is the first *complete* measure.
Some measures are part of the score but are not bars and so have no number of
their own — an anacrusis, the upbeat written after a repeat barline, an unbarred
cadenza. Their `measure_number` is null and `measure_role` says which kind they
are; `measure_belongs_to` gives the bar they are reported against, which for an
upbeat is the bar it leads into. Cite that bar, and say "the upbeat to m. N"
when a passage begins on one — never present a measure without a number as
though it had one. Always cite printed numbering, never `measure_index`, which
is an internal 0-based position counting every measure and will not match the
user's score. When the user names a measure, they mean the printed number too.
The `symbolic_evidence` JSON is score-derived evidence; analysis values labelled
as candidates are not definitive claims. Do not assert a musical fact that is
not supported by the supplied evidence. If the retrieved material doesn't cover
the question, say so clearly."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def _build_context(results: dict) -> str:
    context_parts = []
    included_measures: set[tuple[int, int]] = set()

    for seg in results["segments"]:
        evidence = get_measure_evidence(
            seg["work_id"], seg["measure_start"], seg["measure_end"]
        )
        # Segments can overlap. Limit the context by unique measures so a
        # broad retrieval remains readable and economical for the LLM.
        unique_evidence = []
        for measure in evidence:
            identity = (seg["work_id"], measure["measure_index"])
            if identity in included_measures:
                continue
            if len(included_measures) >= MAX_SYMBOLIC_CONTEXT_MEASURES:
                break
            included_measures.add(identity)
            unique_evidence.append(measure)
        context_parts.append(
            f"[Score excerpt] {seg['composer']} — {seg['title']} "
            f"({seg.get('opus','')}) "
            f"mm. {seg['measure_start']}–{seg['measure_end']}: "
            f"{seg['summary_text']}\n"
            f"symbolic_evidence={json.dumps(unique_evidence, ensure_ascii=False, separators=(',', ':'))}"
        )

    for ts in results["text_sources"]:
        context_parts.append(
            f"[{ts['source_type'].capitalize()}] {ts['composer']} — {ts['title']}: "
            f"{ts['content'][:500]}"
        )

    return "\n\n".join(context_parts) or "No relevant passages found."


def _placeholder_answer(query: str, results: dict) -> str:
    answer = f"**(ScoreChat Assistant)**\n\nBased on the retrieved score segments for *\"{query}\"*:\n\n"
    if results["segments"]:
        for i, seg in enumerate(results["segments"]):
            work_title = f"{seg['composer']} — {seg['title']}"
            if seg.get('opus'):
                work_title += f" ({seg['opus']})"
            answer += f"### {i+1}. {work_title}, mm. {seg['measure_start']}–{seg['measure_end']}\n"
            answer += f"- **Key & Harmony**: This section is in `{seg['local_key']}`. "
            if seg.get('roman_numerals'):
                answer += f"It features the progression: *{seg['roman_numerals']}*.\n"
            else:
                answer += "The harmonic structure is undefined.\n"
            answer += f"- **Texture & Rhythm**: It has a `{seg['texture_tag']}` texture with a `{seg['harmonic_rhythm']}` harmonic rhythm.\n"
            answer += f"- **Contextual Summary**: *{seg['summary_text']}*\n\n"
    else:
        answer += "No relevant score segments were found in the database. Please ensure you have ingested some scores!"
    return answer


def chat(
    query: str,
    composer: str | None = None,
    local_key: str | None = None,
    formal_function: str | None = None,
    texture_tag: str | None = None,
    top_k: int = 6,
    model: str | None = None,
) -> dict:
    """
    Full RAG query. Returns:
        {
          "answer": str,
          "segments": [...],    # retrieved score segments (for UI rendering)
          "text_sources": [...]
        }
    """
    results = retrieve(
        query=query,
        composer=composer,
        local_key=local_key,
        formal_function=formal_function,
        texture_tag=texture_tag,
        top_k=top_k,
    )

    if not chat_provider_ready():
        return {
            "answer":       _placeholder_answer(query, results),
            "segments":     results["segments"],
            "text_sources": results["text_sources"],
        }

    llm = get_chat_model(model=model, temperature=0.3)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": _build_context(results), "question": query})

    return {
        "answer":       answer,
        "segments":     results["segments"],
        "text_sources": results["text_sources"],
    }
