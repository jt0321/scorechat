"""
index_scores.py
---------------
Ingests PDF and .txt files from ./data/ into a local FAISS vector store.
Run once before launching scorechat_app.py, and re-run whenever you add new files.

Usage:
    python index_scores.py
"""

from pathlib import Path
from dotenv import load_dotenv
import os

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = Path("data")
INDEX_DIR = Path("faiss_index")


def extract_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def load_corpus() -> list[str]:
    texts = []
    for p in sorted(DATA_DIR.glob("*.pdf")):
        print(f"  PDF  \u2192 {p.name}")
        t = extract_pdf(p)
        if t.strip():
            texts.append(f"[SOURCE: {p.name}]\n\n{t}")
    for p in sorted(DATA_DIR.glob("*.txt")):
        print(f"  TXT  \u2192 {p.name}")
        t = p.read_text(encoding="utf-8", errors="ignore")
        if t.strip():
            texts.append(f"[SOURCE: {p.name}]\n\n{t}")
    return texts


def main():
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    print("Scanning ./data/ ...")
    texts = load_corpus()

    if not texts:
        print(
            "No PDFs or .txt files found in ./data/\n"
            "Drop your IMSLP score PDFs there and run again."
        )
        return

    full_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(full_text)
    print(f"Created {len(chunks)} chunks from {len(texts)} document(s).")

    print("Embedding and building FAISS index ...")
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    vs = FAISS.from_texts(chunks, embeddings)
    vs.save_local(str(INDEX_DIR))
    print(f"Index saved to ./{INDEX_DIR}/  \u2713")


if __name__ == "__main__":
    main()
