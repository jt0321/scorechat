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

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

load_dotenv()

INDEX_DIR = Path("faiss_index")


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
