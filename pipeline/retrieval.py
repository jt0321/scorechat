"""
pipeline/retrieval.py
Hybrid retrieval: structured metadata filters + pgvector cosine similarity,
exposed as a LangChain BaseRetriever so it composes with LCEL chains.
Returns ranked score segment and text source chunks for a natural-language query.
"""

from __future__ import annotations
from typing import Optional
from sqlalchemy import text
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from db.session import get_session
from pipeline.embedder import embed_single


class HybridScoreRetriever(BaseRetriever):
    """
    Retrieves score_segments + text_sources via pgvector cosine similarity,
    optionally narrowed by structured filters on the score segments.
    """

    composer: Optional[str] = None
    local_key: Optional[str] = None
    formal_function: Optional[str] = None
    texture_tag: Optional[str] = None
    top_k: int = 8
    model: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        session = get_session()
        query_vec = embed_single(query, model=self.model)

        segment_filters = ["1=1"]
        params: dict = {"vec": str(query_vec), "k": self.top_k}

        if self.composer:
            segment_filters.append("w.composer ILIKE :composer")
            params["composer"] = f"%{self.composer}%"
        if self.local_key:
            segment_filters.append("ss.local_key ILIKE :local_key")
            params["local_key"] = f"%{self.local_key}%"
        if self.formal_function:
            segment_filters.append("ss.formal_function = :formal_function")
            params["formal_function"] = self.formal_function
        if self.texture_tag:
            segment_filters.append("ss.texture_tag = :texture_tag")
            params["texture_tag"] = self.texture_tag

        where_clause = " AND ".join(segment_filters)

        segment_sql = text(f"""
            SELECT
                ss.id,
                w.composer,
                w.title,
                w.opus,
                ss.measure_start,
                ss.measure_end,
                ss.local_key,
                ss.roman_numerals,
                ss.harmonic_rhythm,
                ss.texture_tag,
                ss.formal_function,
                ss.summary_text,
                ss.musicxml_slice,
                1 - (ss.embedding <=> CAST(:vec AS vector)) AS cosine_similarity
            FROM score_segments ss
            JOIN works w ON w.id = ss.work_id
            WHERE {where_clause}
            ORDER BY ss.embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)
        seg_rows = session.execute(segment_sql, params).mappings().all()

        # --- Text sources: vector search (no structural filters needed) ---
        text_sql = text("""
            SELECT
                ts.id,
                w.composer,
                w.title,
                ts.source_type,
                ts.content,
                ts.url,
                1 - (ts.embedding <=> CAST(:vec AS vector)) AS cosine_similarity
            FROM text_sources ts
            JOIN works w ON w.id = ts.work_id
            ORDER BY ts.embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)
        text_rows = session.execute(text_sql, {"vec": str(query_vec), "k": self.top_k}).mappings().all()

        documents = []
        for r in seg_rows:
            row = dict(r)
            documents.append(
                Document(page_content=row.get("summary_text") or "", metadata={**row, "kind": "segment"})
            )
        for r in text_rows:
            row = dict(r)
            documents.append(
                Document(page_content=row.get("content") or "", metadata={**row, "kind": "text_source"})
            )
        return documents


def retrieve(
    query: str,
    composer: Optional[str] = None,
    local_key: Optional[str] = None,
    formal_function: Optional[str] = None,
    texture_tag: Optional[str] = None,
    top_k: int = 8,
    model: Optional[str] = None,
) -> dict:
    """
    Hybrid retrieval over score_segments and text_sources, via HybridScoreRetriever.

    Returns:
        {
          "segments": [...],   # list of score_segment rows
          "text_sources": [...] # list of text_source rows
        }
    """
    retriever = HybridScoreRetriever(
        composer=composer,
        local_key=local_key,
        formal_function=formal_function,
        texture_tag=texture_tag,
        top_k=top_k,
        model=model,
    )
    documents = retriever.invoke(query)

    segments = []
    text_sources = []
    for doc in documents:
        row = {k: v for k, v in doc.metadata.items() if k != "kind"}
        if doc.metadata.get("kind") == "segment":
            segments.append(row)
        else:
            text_sources.append(row)

    return {"segments": segments, "text_sources": text_sources}
