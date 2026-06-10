"""
pipeline/chat.py
Scorechat RAG: retrieve score segments + text sources, then generate
a grounded LLM response with measure-level citations.
"""

from __future__ import annotations
import os
from openai import OpenAI
from pipeline.retrieval import retrieve

SYSTEM_PROMPT = """You are Scorechat, an expert musicologist and piano pedagogue.
Answer questions about musical scores using the retrieved score excerpts and
reference materials provided below. Cite specific measure ranges when discussing
musical passages (e.g. "mm. 17–20"). Be precise about harmony, texture, and form.
If the retrieved material doesn't cover the question, say so clearly."""


def chat(
    query: str,
    composer: str | None = None,
    local_key: str | None = None,
    formal_function: str | None = None,
    texture_tag: str | None = None,
    top_k: int = 6,
    model: str = "gpt-4o",
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

    # Build retrieval context for LLM
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

    context = "\n\n".join(context_parts) or "No relevant passages found."

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
    )

    return {
        "answer":       response.choices[0].message.content,
        "segments":     results["segments"],
        "text_sources": results["text_sources"],
    }
