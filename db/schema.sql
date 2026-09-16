-- Works: one row per musical work
CREATE TABLE works (
    id              SERIAL PRIMARY KEY,
    composer        TEXT NOT NULL,
    title           TEXT NOT NULL,
    opus            TEXT,
    nickname        TEXT,         -- e.g. "Moonlight", "Appassionata"
    catalog_no      TEXT,         -- e.g. K.331, BWV 772
    work_number     INT,          -- e.g. 14 (Piano Sonata No. 14)
    movement_number INT,          -- e.g. 3 (third movement)
    tempo_indication TEXT,        -- e.g. "Presto agitato" — identifies the movement
    key_signature   TEXT,         -- e.g. "A major"
    time_signature  TEXT,
    year_composed   INT,
    instrumentation TEXT DEFAULT 'solo piano',
    imslp_url       TEXT,
    wikipedia_url   TEXT,
    source_license  TEXT DEFAULT 'public domain',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Score assets: the Humdrum source and the MEI rendered from it
CREATE TABLE score_assets (
    id          SERIAL PRIMARY KEY,
    work_id     INT REFERENCES works(id) ON DELETE CASCADE,
    asset_type  TEXT NOT NULL CHECK (asset_type IN ('krn','mei')),
    file_path   TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Immutable provenance and raw symbolic source.  Parsed/analysed records are
-- reproducible derivatives; the .krn content retained here is authoritative.
CREATE TABLE score_sources (
    id          SERIAL PRIMARY KEY,
    work_id     INT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    format      TEXT NOT NULL DEFAULT 'humdrum-kern',
    file_path   TEXT NOT NULL,
    source_url  TEXT,
    sha256      TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (work_id, sha256)
);

-- Canonical JSON-safe encoding of every notated measure across all parts.
CREATE TABLE score_measures (
    id             SERIAL PRIMARY KEY,
    work_id        INT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    measure_index  INT NOT NULL,
    -- NULL when the measure is not a bar: an anacrusis, the upbeat written
    -- after a repeat barline, an unbarred passage. measure_role says which,
    -- and measure_belongs_to the printed bar it is reported against.
    measure_number INT,
    measure_role   TEXT NOT NULL DEFAULT 'bar'
                   CHECK (measure_role IN ('bar','anacrusis','upbeat','unbarred','empty')),
    measure_belongs_to INT,
    CHECK ((measure_role = 'bar') = (measure_number IS NOT NULL)),
    symbolic_data  JSONB NOT NULL,
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (work_id, measure_index)
);

-- Versioned deterministic analysis derived from one canonical score measure.
CREATE TABLE measure_analyses (
    id               SERIAL PRIMARY KEY,
    measure_id       INT NOT NULL REFERENCES score_measures(id) ON DELETE CASCADE,
    analysis_version TEXT NOT NULL,
    analysis_data    JSONB NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (measure_id, analysis_version)
);

-- Provenance for a reproducible pass that creates broader score analyses.
CREATE TABLE analysis_runs (
    id                 SERIAL PRIMARY KEY,
    work_id            INT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    analyzer_name      TEXT NOT NULL,
    analyzer_version   TEXT NOT NULL,
    configuration_data JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_sha256      TEXT NOT NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Variable-length, evidence-backed spans. Initially these are deterministic
-- boundary candidates, not claims about phrase, theme, or variation.
CREATE TABLE span_analyses (
    id              SERIAL PRIMARY KEY,
    work_id         INT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
    analysis_run_id INT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    measure_start_index INT NOT NULL,
    measure_end_index   INT NOT NULL,
    measure_start   INT NOT NULL,
    measure_end     INT NOT NULL,
    span_type       TEXT NOT NULL DEFAULT 'candidate'
                    -- 'section' is notated structure read from Humdrum
                    -- expansion records; the rest are derived analyses.
                    CHECK (span_type IN ('candidate','section','phrase','theme','variation','transition')),
    label           TEXT,
    confidence      REAL,
    status          TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('proposed','accepted','rejected')),
    evidence_data   JSONB NOT NULL,
    features_data   JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    CHECK (measure_start_index <= measure_end_index)
);

-- Relations are reserved for validated symbolic comparisons between spans.
CREATE TABLE span_relations (
    id              SERIAL PRIMARY KEY,
    analysis_run_id INT NOT NULL REFERENCES analysis_runs(id) ON DELETE CASCADE,
    source_span_id  INT NOT NULL REFERENCES span_analyses(id) ON DELETE CASCADE,
    target_span_id  INT NOT NULL REFERENCES span_analyses(id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL CHECK (relation_type IN
                    ('repeats','varies','inverts','diminishes','augments','changes_meter_from','contrasts_with')),
    confidence      REAL,
    status          TEXT NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('proposed','accepted','rejected')),
    evidence_data   JSONB NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX ON score_measures (work_id, measure_index);
CREATE INDEX ON measure_analyses (measure_id);
CREATE INDEX ON analysis_runs (work_id, created_at);
CREATE INDEX ON span_analyses (work_id, measure_start_index, measure_end_index);
CREATE INDEX ON span_analyses (analysis_run_id);
CREATE INDEX ON span_relations (source_span_id, target_span_id);
CREATE INDEX ON works (composer);

-- Full-text metadata index (composer/title/opus/nickname/tempo) — lets a query
-- like "the Moonlight sonata", "Op. 111" or "Presto agitato" be matched against
-- work identity directly, which is what resolve_work does before anything else.
CREATE INDEX works_metadata_fts_idx ON works USING gin (
    to_tsvector('english',
        coalesce(composer, '') || ' ' ||
        coalesce(title, '') || ' ' ||
        coalesce(opus, '') || ' ' ||
        coalesce(nickname, '') || ' ' ||
        coalesce(tempo_indication, '')
    )
);
