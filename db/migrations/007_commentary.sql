-- 007: commentary -- published writing about the corpus, anchored to works.
--
-- The texts themselves are not stored in the repository (see
-- sources/manifest.toml); these tables hold what was read from them, which
-- is rebuilt from the fetched files by ingest_commentary.py.
--
-- Commentary is attributed opinion, never musical fact: nothing here is read
-- by the symbolic passes, and a passage's sonata and movements are anchors for
-- finding it, not claims about the score.

-- Only prose is embedded -- never score content. See passage_embeddings.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS text_documents (
    id            SERIAL PRIMARY KEY,
    source_key    TEXT NOT NULL UNIQUE,      -- the key in sources/manifest.toml
    title         TEXT NOT NULL,
    author        TEXT NOT NULL,
    translator    TEXT,
    year          INT,
    citation      TEXT NOT NULL,
    archive_id    TEXT,
    rights        TEXT NOT NULL,
    file_sha256   JSONB NOT NULL,            -- {file: sha256} the passages were read from
    ingested_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS text_passages (
    id                SERIAL PRIMARY KEY,
    document_id       INT NOT NULL REFERENCES text_documents(id) ON DELETE CASCADE,
    ordinal           INT NOT NULL,          -- reading order within the document
    section_heading   TEXT NOT NULL,
    declared_sonatas  INT[] NOT NULL DEFAULT '{}',   -- what the manifest says the section covers
    sonata            INT,                   -- works.work_number the passage is anchored to
    work_ids          INT[] NOT NULL DEFAULT '{}',   -- movements it discusses; empty = the sonata as a whole
    anchor_status     TEXT NOT NULL
                      CHECK (anchor_status IN ('confirmed','declared','conflict','named','unanchored')),
    anchor_evidence   JSONB NOT NULL DEFAULT '{}',
    leaf              INT,                   -- scan page (archive.org …/page/n{leaf})
    page              INT,                   -- printed page
    line              INT,                   -- plain-text sources have lines, not pages
    content           TEXT NOT NULL,
    content_tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (document_id, ordinal),
    -- Only a declared section can confirm, merely declare, or conflict.
    CHECK (anchor_status IN ('named','unanchored') OR cardinality(declared_sonatas) > 0)
);
CREATE INDEX IF NOT EXISTS text_passages_tsv_idx ON text_passages USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS text_passages_sonata_idx ON text_passages (sonata);
CREATE INDEX IF NOT EXISTS text_passages_work_ids_idx ON text_passages USING gin (work_ids);

-- One row per passage per embedding model. The column has no fixed dimension,
-- so several models can coexist: an embedding is a property of the corpus as
-- embedded by one model, and a query must be embedded by the same one. The
-- corpus is small enough (under a thousand passages) that search is an exact
-- scan, so there is no index -- and no dimension lock of the kind that made the
-- old retrieval layer costly to change.
CREATE TABLE IF NOT EXISTS passage_embeddings (
    passage_id  INT NOT NULL REFERENCES text_passages(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,               -- "provider:model", e.g. "gemini:gemini-embedding-2"
    dimensions  INT NOT NULL,
    embedding   vector NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (passage_id, model),
    CHECK (vector_dims(embedding) = dimensions)
);
