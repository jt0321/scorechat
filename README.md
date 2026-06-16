# ScoreChat — Classical Score RAG

![ScoreChat Logo](scorechat_ui_mockup.png)

ScoreChat is a Retrieval-Augmented Generation (RAG) system that allows you to chat directly with classical piano scores. It ingests public-domain sheet music (via Humdrum `.krn` or standard MusicXML), extracts musical features (local keys, Roman numerals, texture, and harmonic rhythm) using `music21`, encodes them as vector embeddings, and stores them in PostgreSQL (`pgvector`). The assistant answers musical queries grounded in the actual notation with measure-level citations and interactive SVG notation rendering.

---

## Features

- **High-Quality Symbolic Ingestion**: Supports direct ingestion of Humdrum (`.krn`) and MuseScore (`.mscx`) files, automatically converting them to standardized `.musicxml` files via `music21`.
- **Musical Feature Analysis**: Automatically partitions scores into measure-level chunks, extracting local key signatures, Roman numeral progressions, harmonic rhythm, and texture classifications.
- **WASM Notation Rendering**: Generates MEI (`.mei`) files via `verovio`, allowing the frontend to dynamically render exact SVG notation slices of the retrieved measures.
- **Hybrid Vector Retrieval**: Combines pgvector cosine similarity search on musical/analytical summaries with traditional metadata filters.
- **Double Interface**: Offer both a clean **Streamlit chatbot** and a customized **HTML/JS frontend** served via a Python HTTP server.

---

## Quickstart

### 1. Set Up Environment
Create a virtual environment and install the required dependencies (requires `uv` for speed):
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2. Launch the Vector Database
Launch the local PostgreSQL database preloaded with `pgvector` (requires Docker):
```bash
docker compose up -d
```

### 3. Add API Keys
Copy the example environment file and add your `OPENAI_API_KEY`:
```bash
cp .env.example .env
# Edit .env to add your API key
```

### 4. Ingest Repertoire
Download public-domain PDFs from IMSLP and push the scores through the database pipeline:
```bash
# Download PDFs
python download_scores.py

# Ingest and convert pre-existing symbolic scores (converting .krn -> .musicxml -> .mei)
python ingest_scores.py --mei
```

### 5. Launch the Web Interface
You can run either of the two user interfaces:

* **HTML/JS Client & API Server**:
  ```bash
  python server.py
  ```
  Then open [http://localhost:8000](http://localhost:8000) in your browser.
  
* **Streamlit Chatbot**:
  ```bash
  streamlit run scorechat_app.py
  ```

---

## Project Structure

```
scorechat/
├── data/
│   ├── beethoven_piano_sonata_no_32_in_c_minor.krn      # Raw Humdrum score (ground truth)
│   ├── beethoven_piano_sonata_no_32_in_c_minor.musicxml # Auto-generated standard MusicXML
│   └── mei/
│       └── beethoven_piano_sonata_no_32_in_c_minor.mei  # Auto-generated MEI for SVG rendering
├── db/
│   ├── models.py          # SQLAlchemy models for Works, Assets, and Segments
│   ├── schema.sql         # SQL schema definitions for pgvector tables
│   └── store.py           # Database persistence and cleanup functions
├── ingest/
│   └── omr.py             # Fitz-based PDF page rendering and OMR fallbacks (oemer/audiveris)
├── analysis/
│   └── analyzer.py        # music21-based musical feature extraction and segmentation
├── pipeline/
│   ├── chat.py            # RAG chat logic and LLM prompt framing
│   ├── retrieval.py       # pgvector cosine similarity score search
│   ├── embedder.py        # OpenAI text embeddings generator
│   └── mei_converter.py   # Verovio-based MusicXML-to-MEI converter
├── frontend/
│   ├── index.html         # Custom HTML/JS chat client and notation viewer
│   └── score_viewer.html  # Standalone score rendering panel
├── download_scores.py     # Pulls public-domain PDFs from IMSLP
├── ingest_scores.py       # Batch ingests all files in data/ to postgres
├── server.py              # API server and static host for the HTML/JS client
└── scorechat_app.py       # Alternative Streamlit chatbot UI
```

---

## Note on OMR (Optical Music Recognition)
PDF-to-symbolic conversion using OMR tools (like `oemer`) is highly error-prone on complex, multi-page classical piano music (e.g., late Beethoven sonatas). For data integrity, the ScoreChat pipeline prioritizes **pre-existing symbolic scores** (`.krn` or `.musicxml`) placed in `data/`, automatically converting them to MusicXML and MEI, and bypassing the error-prone OMR step.
