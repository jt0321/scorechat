"""
commentary/search.py
Finding passages of commentary: full-text and vector search, fused.

Two retrievers, and every result says which found it. Full-text search needs
no provider and always runs; it is the baseline anything cleverer has to beat.
Vector search runs when the corpus has been embedded with a model, and the
query is embedded by *that same* model -- vectors from different models are not
comparable, which is why the model is a property of the corpus, not a
per-question choice. The two rankings are combined by reciprocal rank fusion,
which needs no tuning between scores on different scales.

Filters are anchors, not relevance: asking about a movement returns passages
recorded against that movement, and also passages about its sonata as a whole,
marked as such -- a general remark about the Moonlight is still about its
finale in part, but it should not be mistaken for a remark on the finale alone.
"""

from __future__ import annotations

import os
import re

from sqlalchemy import text

from db.session import session_scope

RRF_K = 60            # the standard constant; larger flattens rank differences
CANDIDATES = 30       # how deep each retriever looks before fusion

_STOPWORDS = {"the", "and", "for", "with", "what", "does", "about", "that", "this", "which",
              "who", "how", "say", "says", "said", "are", "was", "were", "his", "her", "its",
              "from", "into", "any", "there", "their", "they", "them", "than", "then", "when"}


def embedded_models() -> list[dict]:
    with session_scope() as session:
        rows = session.execute(text("""
            SELECT model, dimensions, count(*) AS passages
            FROM passage_embeddings GROUP BY model, dimensions ORDER BY count(*) DESC""")).mappings()
        return [dict(r) for r in rows]


def default_model() -> str | None:
    """COMMENTARY_EMBEDDING_MODEL if set, else the model most of the corpus is
    embedded with, else none -- full-text search alone."""
    from pipeline.providers import embedding_model_id
    configured = os.environ.get("COMMENTARY_EMBEDDING_MODEL")
    if configured:
        return embedding_model_id(configured)
    models = embedded_models()
    return models[0]["model"] if models else None


def _or_query(query: str) -> str:
    """Any of the question's words, not all of them: a musician's question
    shares a few words with the passage that answers it, rarely every one."""
    words = [w for w in re.findall(r"[A-Za-z]{3,}", query) if w.lower() not in _STOPWORDS]
    return " | ".join(dict.fromkeys(w.lower() for w in words))


def _filters(sonata: int | None, work_id: int | None) -> tuple[str, dict]:
    clauses, params = [], {}
    if sonata is not None:
        clauses.append("(p.sonata = :sonata OR :sonata = ANY(p.declared_sonatas))")
        params["sonata"] = sonata
    if work_id is not None:
        clauses.append("""(:work_id = ANY(p.work_ids)
                           OR (cardinality(p.work_ids) = 0
                               AND p.sonata = (SELECT work_number FROM works WHERE id = :work_id)))""")
        params["work_id"] = work_id
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def _fulltext(session, query: str, where: str, params: dict) -> list[int]:
    terms = _or_query(query)
    if not terms:
        return []
    rows = session.execute(text(f"""
        SELECT p.id FROM text_passages p, to_tsquery('english', :terms) q
        WHERE p.content_tsv @@ q {where}
        ORDER BY ts_rank_cd(p.content_tsv, q) DESC, p.id LIMIT :k"""),
        {**params, "terms": terms, "k": CANDIDATES})
    return [r[0] for r in rows]


def _vector(session, vector: list[float], model: str, where: str, params: dict) -> list[int]:
    rows = session.execute(text(f"""
        SELECT p.id FROM passage_embeddings e JOIN text_passages p ON p.id = e.passage_id
        WHERE e.model = :model {where}
        ORDER BY e.embedding <=> CAST(:vector AS vector), p.id LIMIT :k"""),
        {**params, "model": model, "vector": "[" + ",".join(map(str, vector)) + "]", "k": CANDIDATES})
    return [r[0] for r in rows]


def search_passages(query: str, *, sonata: int | None = None, work_id: int | None = None,
                    model: str | None = None, limit: int = 5, embed=None) -> dict:
    """Passages answering `query`, best first, each with its citation and anchor.

    `embed` embeds the query; it defaults to the provider behind `model`, and
    is a parameter so the search can be tested without one.
    """
    from pipeline.providers import embedding_model_id, embedding_provider_ready, get_embeddings
    model = embedding_model_id(model) if model else default_model()
    where, params = _filters(sonata, work_id)
    note = None

    with session_scope() as session:
        rankings = {"fulltext": _fulltext(session, query, where, params)}
        if model:
            available = {m["model"] for m in embedded_models()}
            if model not in available:
                note = f"The commentary has not been embedded with {model}; full-text search only."
            elif embed is None and not embedding_provider_ready(model):
                note = f"No API key for {model.split(':')[0]}; full-text search only."
            else:
                vector = (embed or get_embeddings(model).embed_query)(query)
                rankings["vector"] = _vector(session, vector, model, where, params)

        fused: dict[int, float] = {}
        for ranking in rankings.values():
            for rank, passage_id in enumerate(ranking, start=1):
                fused[passage_id] = fused.get(passage_id, 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(fused, key=lambda i: (-fused[i], i))[:limit]
        passages = _load(session, best) if best else {}

    results = []
    for passage_id in best:
        row = passages[passage_id]
        row["found_by"] = {name: ranking.index(passage_id) + 1
                           for name, ranking in rankings.items() if passage_id in ranking}
        row["scope"] = ("movement" if work_id is not None and work_id in row["work_ids"]
                        else "sonata" if row["sonata"] is not None else "general")
        results.append(row)
    if not results and note is None:
        note = "No passage of the commentary matches; the texts may simply not discuss this."
    return {"query": query, "retrievers": list(rankings), "model": model if "vector" in rankings else None,
            "passages": results, "note": note}


def _load(session, ids: list[int]) -> dict[int, dict]:
    rows = session.execute(text("""
        SELECT p.id, p.content, p.page, p.leaf, p.line, p.sonata, p.work_ids, p.anchor_status,
               p.section_heading, d.author, d.title, d.year, d.citation, d.archive_id, d.source_key
        FROM text_passages p JOIN text_documents d ON d.id = p.document_id
        WHERE p.id = ANY(:ids)"""), {"ids": ids}).mappings()
    out = {}
    for r in rows:
        row = dict(r)
        row["work_ids"] = list(row["work_ids"] or [])
        row["url"] = (f"https://archive.org/details/{row['archive_id']}/page/n{row['leaf']}"
                      if row["leaf"] is not None else f"https://archive.org/details/{row['archive_id']}")
        out[row["id"]] = row
    return out
