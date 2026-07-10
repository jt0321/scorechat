"""
pipeline/chat.py
Scorechat RAG: retrieve score segments + text sources, then generate
a grounded LLM response with measure-level citations via a LangChain
LCEL chain (prompt | chat model | StrOutputParser). The chat model
backend is selected by CHAT_PROVIDER (see pipeline/providers.py) so
this isn't locked to OpenAI.
"""

from __future__ import annotations
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pipeline.providers import get_chat_model, chat_provider_ready
from pipeline.retrieval import retrieve

SYSTEM_PROMPT = """You are Scorechat, an expert musicologist and piano pedagogue.
Answer questions about musical scores using the retrieved score excerpts and
reference materials provided below. Cite specific measure ranges when discussing
musical passages (e.g. "mm. 17–20"). Be precise about harmony, texture, and form.
If the retrieved material doesn't cover the question, say so clearly."""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def _build_context(results: dict) -> str:
    context_parts = []

    for seg in results["segments"]:
        context_parts.append(
            f"[Score excerpt] {seg['composer']} — {seg['title']} "
            f"({seg.get('opus','')}) "
            f"mm. {seg['measure_start']}–{seg['measure_end']}: "
            f"{seg['summary_text']}"
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
