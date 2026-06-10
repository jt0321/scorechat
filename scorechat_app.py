"""
scorechat_app.py
----------------
Streamlit chat interface for ScoreChat.
Requires a populated pgvector database (run ingest_scores.py first).

Usage:
    streamlit run scorechat_app.py
"""

import csv
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from pipeline.chat import chat

load_dotenv()

FRSM_CSV = Path("data/frsm_scores.csv")


@st.cache_data(show_spinner=False)
def load_frsm_scores() -> dict[str, list[dict]]:
    """Load FRSM score list from CSV; return dict keyed by composer."""
    if not FRSM_CSV.exists():
        return {}
    scores: dict[str, list[dict]] = defaultdict(list)
    with open(FRSM_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[row["composer"]].append(row)
    return dict(scores)


@st.cache_resource(show_spinner="Connecting to pipeline...")
def _warmup():
    return chat


def main():
    st.set_page_config(page_title="ScoreChat", page_icon="\U0001f3bc", layout="wide")
    st.title("\U0001f3bc ScoreChat")
    st.caption("Classical Score RAG — ask questions grounded in your uploaded scores.")

    with st.sidebar:
        st.header("FRSM Repertoire")
        frsm = load_frsm_scores()
        if frsm:
            for composer in sorted(frsm.keys()):
                with st.expander(composer):
                    for row in frsm[composer]:
                        label = row["work"]
                        if row.get("nickname"):
                            label += f" “{row['nickname']}”"
                        if row.get("opus"):
                            label += f", Op. {row['opus'].lstrip('Op. ')}"
                        elif row.get("catalog"):
                            label += f", {row['catalog']}"
                        if row.get("imslp_url"):
                            st.markdown(f"- [{label}]({row['imslp_url']})")
                        else:
                            st.markdown(f"- {label}")
        else:
            st.caption(
                "No FRSM CSV found. "
                "Run `python download_scores.py --csv data/frsm_scores.csv`."
            )

    _warmup()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about a piece, key, form, or bar number...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            result = chat(query)

        answer   = result["answer"]
        segments = result.get("segments", [])

        st.markdown(answer)

        if segments:
            with st.expander("Score excerpts"):
                for seg in segments:
                    st.markdown(
                        f"**{seg.get('composer','')} — {seg.get('title','')} "
                        f"({seg.get('opus','')})  "
                        f"mm. {seg.get('measure_start','')}–{seg.get('measure_end','')}**"
                    )
                    st.write(seg.get("summary_text", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
