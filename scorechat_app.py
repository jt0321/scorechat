"""
scorechat_app.py
----------------
Streamlit chat interface for ScoreChat.
Requires a FAISS index built by index_scores.py.

Usage:
    streamlit run scorechat_app.py
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import csv
from collections import defaultdict

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

INDEX_DIR = Path("faiss_index")
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


@st.cache_resource(show_spinner="Loading index...")
def load_chain():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vs = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )


def main():
    st.set_page_config(page_title="ScoreChat", page_icon="\U0001f3bc", layout="wide")
    st.title("\U0001f3bc ScoreChat")
    st.caption("Classical Score RAG \u2014 ask questions grounded in your uploaded scores.")

    # --- Sidebar: FRSM score list ---
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
                            label += f", Op. {row['opus'].lstrip('Op. ')}"
                        elif row.get("catalog"):
                            label += f", {row['catalog']}"
                        if row.get("imslp_url"):
                            st.markdown(f"- [{label}]({row['imslp_url']})")
                        else:
                            st.markdown(f"- {label}")
        else:
            st.caption("No FRSM CSV found. Run `python download_scores.py --csv data/frsm_scores.csv`.")

    if not INDEX_DIR.exists():
        st.warning(
            "No FAISS index found. "
            "Add PDFs/txt to `./data/` and run `python index_scores.py` first."
        )
        st.stop()

    chain = load_chain()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask about a piece, key, form, or bar number...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving..."):
                result = chain.invoke({"query": query})
            answer = result["result"]
            sources = result.get("source_documents", [])

            st.markdown(answer)

            if sources:
                with st.expander("Source chunks"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}**")
                        st.write(doc.page_content[:600] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
