"""
pipeline/retrieval.py
Hybrid retrieval: structured metadata filters + pgvector cosine similarity.
Returns ranked score segment and text source chunks for a natural-language query.
"""

from __future__ import annotations
from typing import Optional
from sqlalchemy import text
from db.session import get_session
from pipeline.embedder import embed_single


def retrieve(
    query: str,
    composer: Optional[str]    = None,
    local_key: Optional[str]   = None,
    formal_function: Optional[str] = None,
    texture_tag: Optional[str] = None,
    top_k: int = 8,
    model: str = "text-embedding-3-small",
) -> dict:
    """
    Hybrid retrieval over score_segments and text_sources.

    Returns:
        {
          "segments": [...],   # list of score_segment rows
          "text_sources": [...] # list of text_source rows
        }
    """
    session = get_session()
    query_vec = embed_single(query, model=model)

    # --- Score segments: vector + optional metadata filters ---
    segment_filters = ["1=1"]
    params: dict = {"vec": str(query_vec), "k": top_k}

    if composer:
        segment_filters.append("w.composer ILIKE :composer")
        params["composer"] = f"%{composer}%"
    if local_key:
        segment_filters.append("ss.local_key ILIKE :local_key")
        params["local_key"] = f"%{local_key}%"
    if formal_function:
        segment_filters.append("ss.formal_function = :formal_function")
        params["formal_function"] = formal_function
    if texture_tag:
        segment_filters.append("ss.texture_tag = :texture_tag")
        params["texture_tag"] = texture_tag

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
            1 - (ss.embedding <=> :vec::vector) AS cosine_similarity
        FROM score_segments ss
        JOIN works w ON w.id = ss.work_id
        WHERE {where_clause}
        ORDER BY ss.embedding <=> :vec::vector
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
            1 - (ts.embedding <=> :vec::vector) AS cosine_similarity
        FROM text_sources ts
        JOIN works w ON w.id = ts.work_id
        ORDER BY ts.embedding <=> :vec::vector
        LIMIT :k
    """)
    text_rows = session.execute(text_sql, {"vec": str(query_vec), "k": top_k}).mappings().all()

    return {
        "segments":     [dict(r) for r in seg_rows],
        "text_sources": [dict(r) for r in text_rows],
    }
