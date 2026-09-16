# ScoreChat

Ask questions about Beethoven's piano sonatas and get answers traceable to the score.

The corpus is 32 sonatas — 103 movements — in Humdrum `**kern`, from
[craigsapp/beethoven-piano-sonatas](https://github.com/craigsapp/beethoven-piano-sonatas).
The score is the source of musical truth: the model chooses which stored facts to
fetch and how to phrase them, and invents none of them.

```mermaid
flowchart TB
  KRN["<b>Humdrum .krn</b><br/>103 movements · 32 sonatas"]

  subgraph readers ["two readers of the same file"]
    M21["<b>music21 — load_score()</b><br/>notes · durations · ties · voices · meter<br/><i>falls back to Verovio/humlib where it<br/>truncates — 4 of 103</i>"]
    HUM["<b>analysis/humdrum.py</b><br/>spine-aware: declared key · sections · dynamics<br/><i>read from the raw source,<br/>bypassing the parse</i>"]
  end

  subgraph canonical ["canonical layer — Postgres"]
    SRC["<b>score_sources</b><br/>raw .krn + SHA-256"]
    MEAS["<b>score_measures</b><br/>symbolic_data · numbering · role"]
    ANAL["<b>measure_analyses</b><br/>versioned: keys · chords · directions"]
  end

  subgraph passes ["derived passes — from the database alone, re-runnable without re-ingesting"]
    NUM["<b>numbering</b><br/>which measures are bars<br/><i>renumber_measures.py</i>"]
    HARM["<b>harmony</b><br/>key trajectory · chords<br/><i>build_harmony.py</i>"]
    SEC["<b>sections</b><br/>engraved repeat scheme<br/><i>build_sections.py</i>"]
    REL["<b>relations</b><br/>repeats / varies between spans<br/><i>build_relations.py</i>"]
  end

  subgraph answering ["answering"]
    API["<b>pipeline/analysis_api.py</b><br/>six plain functions over stored analysis"]
    TOOLS["<b>pipeline/tools.py</b><br/>the six bound as LLM tools — the trace is the citation<br/><i>ask.py · GET /api/ask · web client</i>"]
    MEI["<b>MEI via Verovio</b><br/>exact notation for a cited bar range"]
  end

  KRN --> SRC
  KRN --> M21
  KRN --> HUM
  M21 --> MEAS
  HUM --> MEAS
  MEAS --> ANAL
  MEAS --> NUM
  MEAS --> HARM
  SRC --> SEC
  ANAL --> REL
  NUM --> API
  HARM --> API
  SEC --> API
  REL --> API
  API --> TOOLS
  TOOLS --> MEI

  classDef source fill:#131a22,stroke:#131a22,color:#ffffff
  classDef store fill:#eef1f4,stroke:#c9d2da,color:#131a22
  classDef pass fill:#ffffff,stroke:#34417f,color:#131a22
  classDef answer fill:#ffffff,stroke:#c9d2da,color:#131a22
  class KRN source
  class SRC,MEAS,ANAL store
  class NUM,HARM,SEC,REL pass
  class M21,HUM,API,TOOLS,MEI answer
  style readers fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style canonical fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style passes fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style answering fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
docker compose up -d                 # postgres + pgvector
cp .env.example .env                 # add a provider key

python download_beethoven_piano_sonatas.py   # fetch .krn sources
python ingest_scores.py                      # parse, encode, analyse, store
python build_sections.py && python build_relations.py

python ask.py "where is the recapitulation in the Moonlight finale?"
python server.py                     # viewer + API at localhost:8000
```

An existing database needs the migrations in `db/migrations/` applied in order;
`CLAUDE.md` lists them.

## What it answers

Two shapes of question, both served by the same engine:

```bash
python ask.py "describe mm. 195-210 of the Pathetique's first movement in the context of the whole work"
python ask.py "in op 2 no 2 first movement, where does the opening material return?"
```

Each answer prints the tool calls it was built from. **That trace is the
citation** — a claim with no supporting call under it is one to distrust.

## How the model reaches the corpus

**Tool calling** (`ask.py`, `GET /api/ask`, and the web client) is how an answer
is built. `pipeline/tools.py` binds six functions from
`pipeline/analysis_api.py` — `resolve_work`, `describe_span`,
`find_recurrences`, `compare_spans`, `get_key_plan`, `locate_in_form` — which
read stored analysis directly. **No embeddings are involved.** The client
renders the returned trace as citation cards under the answer, and opens the
score at the first bar range the calls named.

**Retrieval** (`GET /api/chat`, the Streamlit app) is the older path:
`pipeline/chat.py` → `pipeline/retrieval.py` → pgvector cosine search over
`score_segments`. Nothing in `frontend/` uses it any more.

### What is actually vectorised

Only `score_segments.summary_text` — 4,519 prose summaries of measure windows,
embedded to 1536 dimensions. **The score itself is not vectorised.** It is stored
exactly, as JSONB in `score_measures.symbolic_data`, and queried as data.

Neither headline question is a retrieval problem: in "describe mm. x–y" the
measures are *given*, so there is nothing to search for, and "find recurring
material" is an edge in `span_relations` that a vector search over prose cannot
reach. Now that no client depends on it, whether the vector layer stays at all
is an open question rather than an inherited fact.

## Data model

| table | rows | what it holds |
|---|---|---|
| `works` | 103 | one row per movement |
| `score_sources` | 103 | the raw `.krn` and its checksum; every later pass re-reads the source from here |
| `score_measures` | 18,761 | canonical measure encoding, plus printed numbering and measure role |
| `measure_analyses` | 18,761 | versioned facts: keys, chords, directions, counts, texture |
| `span_analyses` | 9,449 | engraved sections and derived candidate spans |
| `span_relations` | 2,258 | `repeats`/`varies` between spans, with transposition and key evidence |
| `score_segments` | 4,519 | the only embedded table — prose summaries for retrieval |

Derived passes (`renumber_measures.py`, `build_harmony.py`, `build_sections.py`,
`build_relations.py`) run from the database alone, so analysis is re-derivable in
seconds without re-parsing a single file.

## Current state

`python evaluate_form.py` scores the pipeline against ten movements whose form is
known independently: **49/50** at a four-bar tolerance.

Known gaps, in rough order of consequence:

- **Theme-and-variation movements return no relations at all.** Six movements
  have none; the matcher's own pruning bound excludes diminution, so a variation
  with twice the note events is never compared. `diminishes`/`augments` exist as
  relation types with no analyser behind them.
- **Key estimation is sticky.** 22 of 103 movements report a single key region;
  the Moonlight finale misses its G♯ minor second group.
- **Nothing ranks recapitulation candidates** — the one eval failure.
- **Bar numbers follow Durand 1915 (Dukas)**, the edition the encodings
  transcribe. An urtext such as Henle numbers first and second endings from the
  same bar, so numbers diverge in the 36 movements with alternate endings.

## Design notes

`CLAUDE.md` carries the rationale — why each rule is the way it is, and what
broke before it was. Worth reading before changing the analysis.

## Layout

```
analysis/     humdrum.py (read the .krn directly) · numbering.py · harmony.py
              sections.py · span_relations.py · analyzer.py (music21)
pipeline/     analysis_api.py · tools.py · chat.py · retrieval.py
              embedder.py · providers.py · mei_converter.py
db/           models.py · schema.sql · store.py · migrations/
evaluation/   ground_truth.json · scoring.py
frontend/     index.html · score_viewer.html
tests/        155 tests
```
