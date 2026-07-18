-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Works: one row per musical work
CREATE TABLE works (
    id              SERIAL PRIMARY KEY,
    composer        TEXT NOT NULL,
    title           TEXT NOT NULL,
    opus            TEXT,
    catalog_no      TEXT,         -- e.g. K.331, BWV 772
    key_signature   TEXT,         -- e.g. "A major"
    time_signature  TEXT,
    year_composed   INT,
    instrumentation TEXT DEFAULT 'solo piano',
    imslp_url       TEXT,
    wikipedia_url   TEXT,
    source_license  TEXT DEFAULT 'public domain',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Score assets: PDFs, page images, Humdrum files
CREATE TABLE score_assets (
    id          SERIAL PRIMARY KEY,
    work_id     INT REFERENCES works(id) ON DELETE CASCADE,
    asset_type  TEXT NOT NULL CHECK (asset_type IN ('pdf','page_image','musicxml','mei','midi','krn')),
    file_path   TEXT NOT NULL,
    page_number INT,
    omr_tool    TEXT,             -- e.g. "oemer", "audiveris"
    omr_quality TEXT,             -- "auto","reviewed","manual"
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Score segments: measure-level chunks (analogue of text paragraphs)
CREATE TABLE score_segments (
    id              SERIAL PRIMARY KEY,
    work_id         INT REFERENCES works(id) ON DELETE CASCADE,
    part            TEXT DEFAULT 'grand_staff', -- 'right_hand','left_hand','grand_staff'
    measure_start   INT NOT NULL,
    measure_end     INT NOT NULL,
    local_key       TEXT,         -- e.g. "e minor"
    roman_numerals  TEXT,         -- serialized Roman numeral analysis string
    harmonic_rhythm TEXT,         -- e.g. "slow", "fast", "mixed"
    texture_tag     TEXT,         -- e.g. "alberti_bass", "cantabile", "octaves"
    formal_function TEXT,         -- e.g. "exposition", "development", "transition"
    motif_tags      TEXT[],       -- array of motif labels
    difficulty      INT CHECK (difficulty BETWEEN 1 AND 10),
    summary_text    TEXT,         -- human-readable chunk summary for embedding
    musicxml_slice  TEXT,         -- raw MusicXML fragment for this segment
    embedding       vector(1536), -- OpenAI text-embedding-3-small
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Text sources: Wikipedia, IMSLP notes, program notes, annotations
CREATE TABLE text_sources (
    id          SERIAL PRIMARY KEY,
    work_id     INT REFERENCES works(id) ON DELETE CASCADE,
    source_type TEXT NOT NULL CHECK (source_type IN ('wikipedia','imslp','program_note','annotation')),
    content     TEXT NOT NULL,
    chunk_index INT NOT NULL DEFAULT 0,  -- paragraph/chunk number within source
    embedding   vector(1536),
    url         TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX ON score_segments USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON text_sources   USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON score_segments (work_id, measure_start, measure_end);
CREATE INDEX ON score_segments (local_key);
CREATE INDEX ON score_segments (formal_function);
CREATE INDEX ON works (composer);
