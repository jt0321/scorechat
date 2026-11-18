# ScoreChat architecture

The full pipeline, from the Humdrum source to a cited answer. Amber nodes are published commentary: kept beside the score, never in it. Everything in the "derived passes" box runs from the database alone and can be re-run without re-ingesting.

[Back to the README](../README.md)

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

  subgraph commentary ["commentary — published opinion, beside the score, never in it"]
    TEXTS["<b>sources/manifest.toml</b><br/>Elterlein 1879 · Marx 1895 · Shedlock 1895<br/><i>fetched, pinned by sha256, not committed</i>"]
    PASS["<b>text_passages</b><br/>anchored to sonata · movement<br/><i>full text + vectors, any provider</i>"]
    CLAIMS["<b>passage_claims</b><br/>keys checked against the score<br/><i>build_claims.py</i>"]
  end

  subgraph answering ["answering"]
    API["<b>pipeline/analysis_api.py</b><br/>six functions over stored analysis<br/>+ two over commentary"]
    TOOLS["<b>pipeline/tools.py</b><br/>the nine bound as LLM tools — the trace is the citation<br/><i>ask.py · GET /api/ask · web client</i>"]
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
  TEXTS --> PASS
  PASS --> CLAIMS
  ANAL -. "engraved + estimated keys" .-> CLAIMS
  PASS --> API
  CLAIMS --> API

  classDef source fill:#131a22,stroke:#131a22,color:#ffffff
  classDef store fill:#eef1f4,stroke:#c9d2da,color:#131a22
  classDef pass fill:#ffffff,stroke:#34417f,color:#131a22
  classDef answer fill:#ffffff,stroke:#c9d2da,color:#131a22
  class KRN source
  class SRC,MEAS,ANAL store
  class NUM,HARM,SEC,REL pass
  class M21,HUM,API,TOOLS,MEI answer
  classDef opinion fill:#f7f3ea,stroke:#b8a77a,color:#131a22
  class TEXTS,PASS,CLAIMS opinion
  style readers fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style canonical fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style passes fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style answering fill:#f5f6f7,stroke:#c9d2da,color:#5a6673
  style commentary fill:#fbf9f4,stroke:#d8cdb0,color:#5a6673
```
