# ScoreChat — Classical Score RAG

Chat with piano scores. Drop public-domain PDFs (IMSLP) into `data/`, index them, and ask questions grounded in the actual notation and text — tempo markings, key signatures, form, bar numbers.

Built with LangChain, FAISS, PyMuPDF, and Streamlit.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Add API key
cp .env.example .env
# edit .env → OPENAI_API_KEY=your-key

# 3. Download sample scores
python download_scores.py

# 4. (Optional) Fetch and match scores from a CSV list
python download_scores.py --csv data/frsm_scores.csv

# 5. Index to FAISS
python index_scores.py

# 6. Launch chat
streamlit run scorechat_app.py
```

---

## Project Structure

```
scorechat/
├── download_scores.py     # Fetches public-domain PDFs from IMSLP; supports CSV-driven lookup
├── index_scores.py        # PDF/txt → chunked embeddings → FAISS index
├── scorechat_app.py       # Streamlit chat UI with FRSM sidebar and source expander
├── data/
│   ├── frsm_scores.csv    # Curated score list with IMSLP URLs
│   └── ...                # Drop additional PDFs/txt here
├── chains/
│   ├── rag_chain.py       # LCEL retrieval-augmented generation chain
│   ├── summarization_chain.py
│   └── extraction_chain.py
├── agents/
│   ├── orchestrator.py    # Multi-agent coordinator
│   ├── research_agent.py  # ReAct agent with web/wiki tools
│   └── data_agent.py      # NL → SQL agent
└── faiss_index/           # Auto-generated, gitignored
```

---

## Stack

| Layer | Technology |
|---|---|
| LLM Framework | LangChain (LCEL) |
| LLM | OpenAI GPT-4o-mini |
| Vector Store | FAISS (local) |
| PDF Parsing | PyMuPDF |
| UI | Streamlit |
| Language | Python 3.11+ |

---

## Example Queries

- *"What is the time signature of the opening of the Waldstein Sonata?"*
- *"How does Schumann use the left hand in Kreisleriana?"*
- *"What key does the Appassionata's development section reach?"*
- *"Describe the form of Liszt's Sonata in B minor."*

---

## Score List

All scores sourced from [IMSLP](https://imslp.org) — public domain only. `data/frsm_scores.csv` includes 41 works across 10 composers. A few examples:

- Beethoven — Piano Sonata No. 23 in F minor, Op. 57 (*Appassionata*)
- Chopin — Ballade No. 4 in F minor, Op. 52
- Liszt — Piano Sonata in B minor, S.178
- Schumann — Kreisleriana, Op. 16

The sidebar in the Streamlit app lists all works with direct IMSLP links.

To download and match scores from the CSV:

```bash
python download_scores.py --csv data/frsm_scores.csv
# add --download to also fetch PDFs for matched rows
```

Add your own PDFs to `data/` and re-run `python index_scores.py` to refresh the index.

---

## Notes

- `faiss_index/` and `.env` are gitignored — never commit API keys or generated indexes.
- Re-index any time you add new scores: `python index_scores.py` overwrites the previous index.
- Typed/engraved PDFs extract well; hand-written or image-only scans will yield little text.
- IMSLP rate-limits batch requests. The downloader sleeps 1.2 s between API calls automatically.
