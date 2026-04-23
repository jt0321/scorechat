"""
RAG Chain (Retrieval-Augmented Generation)
------------------------------------------
Indexes local documents into a FAISS vector store and answers
questions grounded in that knowledge base.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from dotenv import load_dotenv
import os

load_dotenv()

SAMPLE_DOCS = [
    Document(
        page_content="""LangChain is a framework for building applications powered by language models.
        It provides tools for chaining LLM calls, managing memory, using external tools,
        and building agents. Key components include: Chains, Agents, Tools, Memory, and Retrievers.""",
        metadata={"source": "langchain_overview"},
    ),
    Document(
        page_content="""RAG (Retrieval-Augmented Generation) combines a retriever with a generator.
        The retriever finds relevant documents from a vector store. The generator (LLM) uses
        those documents as context to produce grounded answers. This reduces hallucination
        and keeps responses factual.""",
        metadata={"source": "rag_overview"},
    ),
    Document(
        page_content="""Multi-agent systems coordinate multiple specialized AI agents.
        An orchestrator routes tasks to the right agent. Agents can use tools like web search,
        calculators, databases, and APIs. ReAct (Reason + Act) is a popular pattern where
        agents alternate between thinking and taking actions.""",
        metadata={"source": "multi_agent_overview"},
    ),
    Document(
        page_content="""FAISS (Facebook AI Similarity Search) is a library for efficient
        similarity search over dense vectors. It's commonly used as a local vector store
        in RAG pipelines. Alternatives include Chroma, Pinecone, and Weaviate.""",
        metadata={"source": "vector_stores"},
    ),
]


def build_vectorstore(docs: list = None):
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = docs or SAMPLE_DOCS
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


def build_rag_chain(docs: list = None):
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the question using only the context provided.
If the context doesn't contain the answer, say "I don't have that information."

Context:
{context}"""),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str, docs: list = None) -> str:
    chain = build_rag_chain(docs)
    return chain.invoke(question)


if __name__ == "__main__":
    questions = [
        "What is RAG and why does it reduce hallucination?",
        "What vector stores can I use with LangChain?",
        "How does a multi-agent system work?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {ask(q)}\n")
