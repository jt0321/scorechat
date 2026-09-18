"""
commentary/store.py
Persisting commentary passages, and reading the catalogue they anchor to.

A document's passages are replaced wholesale on every ingest: they are derived
from a pinned file by deterministic code, so a re-ingest is a rebuild, not an
update. Replacing them also drops their embeddings (by cascade), since those
belong to passage text that may have changed; the ingest reports how many.
"""

from __future__ import annotations

import json

from sqlalchemy import text

from commentary.anchoring import Catalogue
from commentary.manifest import Source
from db.session import session_scope


def load_catalogue() -> Catalogue:
    with session_scope() as session:
        rows = session.execute(text(
            "SELECT id, work_number, opus, nickname, key_signature, movement_number, "
            "tempo_indication FROM works WHERE work_number IS NOT NULL")).mappings().all()
    return Catalogue.from_rows([dict(r) for r in rows])


def replace_document(source: Source, passages: list[dict]) -> tuple[int, int]:
    """Store a source and its passages, replacing any earlier ingest.

    Returns (document id, embeddings dropped with the old passages).
    """
    with session_scope() as session:
        document_id = session.execute(text("""
            INSERT INTO text_documents (source_key, title, author, translator, year, citation,
                                        archive_id, rights, file_sha256, ingested_at)
            VALUES (:key, :title, :author, :translator, :year, :citation,
                    :archive_id, :rights, CAST(:files AS jsonb), NOW())
            ON CONFLICT (source_key) DO UPDATE SET
                title = EXCLUDED.title, author = EXCLUDED.author, translator = EXCLUDED.translator,
                year = EXCLUDED.year, citation = EXCLUDED.citation, archive_id = EXCLUDED.archive_id,
                rights = EXCLUDED.rights, file_sha256 = EXCLUDED.file_sha256, ingested_at = NOW()
            RETURNING id"""), {
                "key": source.key, "title": source.title, "author": source.author,
                "translator": source.translator, "year": source.year,
                "citation": source.citation(), "archive_id": source.archive_id,
                "rights": source.rights,
                "files": json.dumps({f.name: f.sha256 for f in source.files}),
            }).scalar_one()

        dropped = session.execute(text("""
            SELECT count(*) FROM passage_embeddings e JOIN text_passages p ON p.id = e.passage_id
            WHERE p.document_id = :doc"""), {"doc": document_id}).scalar_one()
        session.execute(text("DELETE FROM text_passages WHERE document_id = :doc"), {"doc": document_id})

        for row in passages:
            session.execute(text("""
                INSERT INTO text_passages (document_id, ordinal, section_heading, declared_sonatas,
                    sonata, work_ids, anchor_status, anchor_evidence, leaf, page, line, content)
                VALUES (:doc, :ordinal, :heading, :declared, :sonata, :work_ids, :status,
                        CAST(:evidence AS jsonb), :leaf, :page, :line, :content)"""),
                {**row, "doc": document_id, "evidence": json.dumps(row["evidence"])})
        session.commit()
    return document_id, dropped
