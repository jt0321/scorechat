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

# 3. Download sample scores (Moonlight Sonata, Chopin etudes, etc.)
python download_scores.py

# 4. Index to FAISS
python index_scores.py

# 5. Launch chat
streamlit run scorechat_app.py
```

---

## Project Structure

```
scorechat/
├── download_scores.py     # Fetches public-domain PDFs from IMSLP
├── index_scores.py        # PDF/txt → chunked embeddings → FAISS index
├── scorechat_app.py       # Streamlit chat UI with source expander
├── chains/
│   ├── rag_chain.py       # LCEL retrieval-augmented generation chain
│   ├── summarization_chain.py
│   └── extraction_chain.py
├── agents/
│   ├── orchestrator.py    # Multi-agent coordinator
│   ├── research_agent.py  # ReAct agent with web/wiki tools
│   └── data_agent.py      # NL → SQL agent
├── data/                  # Drop PDFs/txt here
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

- *"What is the key signature of Moonlight Sonata?"*
- *"Describe the form of Chopin Op.10 No.3."*
- *"What tempo markings appear in this etude?"*
- *"Summarize the structure of the first movement."*

---

## Data Sources

All scores sourced from [IMSLP](https://imslp.org) — public domain only. `download_scores.py` includes:

- Beethoven — Piano Sonata No.14, Op.27 No.2 (*Moonlight*)
- Beethoven — Piano Sonata No.1, Op.2 No.1
- Chopin — Étude Op.10 No.3 (*Tristesse*)
- Chopin — Nocturne Op.9 No.2

Add your own PDFs to `data/` and re-run `python index_scores.py` to refresh the index.

---

## Notes

- `faiss_index/` and `.env` are gitignored — never commit API keys or generated indexes.
- Re-index any time you add new scores: `python index_scores.py` overwrites the previous index.
- Typed/engraved PDFs extract well; hand-written or image-only scans will yield little text.
