"""
db/store.py
Persists works, score assets, score segments (with embeddings),
and text source chunks to Postgres via SQLAlchemy.
"""

from __future__ import annotations
from db.models import Work, ScoreAsset, ScoreSegment, TextSource
from db.session import get_session
from analysis.analyzer import MeasureChunk
from pipeline.embedder import embed_texts


def upsert_work(metadata: dict) -> int:
    """Insert or update a Work record. Returns work.id."""
    session = get_session()
    existing = (
        session.query(Work)
        .filter_by(composer=metadata["composer"], title=metadata["title"])
        .first()
    )
    if existing:
        for k, v in metadata.items():
            setattr(existing, k, v)
        session.commit()
        return existing.id

    work = Work(**metadata)
    session.add(work)
    session.commit()
    return work.id


def store_asset(work_id: int, asset_type: str, file_path: str,
                page_number: int | None = None, omr_tool: str | None = None,
                omr_quality: str = "auto") -> int:
    session = get_session()
    asset = ScoreAsset(
        work_id=work_id, asset_type=asset_type,
        file_path=str(file_path), page_number=page_number,
        omr_tool=omr_tool, omr_quality=omr_quality
    )
    session.add(asset)
    session.commit()
    return asset.id


def store_segments(work_id: int, chunks: list[MeasureChunk]) -> None:
    """Embed all chunk summaries and bulk-insert into score_segments."""
    session = get_session()

    texts = [c.summary_text for c in chunks]
    vectors = embed_texts(texts)

    for chunk, vec in zip(chunks, vectors):
        seg = ScoreSegment(
            work_id         = work_id,
            part            = chunk.part,
            measure_start   = chunk.measure_start,
            measure_end     = chunk.measure_end,
            local_key       = chunk.local_key,
            roman_numerals  = chunk.roman_numerals,
            harmonic_rhythm = chunk.harmonic_rhythm,
            texture_tag     = chunk.texture_tag,
            formal_function = chunk.formal_function,
            motif_tags      = chunk.motif_tags or [],
            summary_text    = chunk.summary_text,
            musicxml_slice  = chunk.musicxml_slice,
            embedding       = vec,
        )
        session.add(seg)

    session.commit()


def store_text_chunks(work_id: int, chunks: list[dict]) -> None:
    """
    chunks: list of dicts with keys: source_type, content, url (optional)
    Embeds each chunk and inserts into text_sources.
    """
    session = get_session()
    texts = [c["content"] for c in chunks]
    vectors = embed_texts(texts)

    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        ts = TextSource(
            work_id     = work_id,
            source_type = chunk["source_type"],
            content     = chunk["content"],
            chunk_index = i,
            embedding   = vec,
            url         = chunk.get("url"),
        )
        session.add(ts)

    session.commit()
