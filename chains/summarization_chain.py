"""
Summarization Chain
-------------------
Map-reduce summarization for long documents.
Splits text into chunks, summarizes each, then combines into a final summary.
"""

from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

MAP_PROMPT = PromptTemplate.from_template("""
Summarize the following text concisely, preserving key facts and data:

{text}

CONCISE SUMMARY:
""")

COMBINE_PROMPT = PromptTemplate.from_template("""
You have been given partial summaries of a longer document.
Combine them into a single cohesive, well-structured summary.
Preserve all important facts, numbers, and conclusions.

PARTIAL SUMMARIES:
{text}

FINAL SUMMARY:
""")


def summarize(text: str, chunk_size: int = 3000) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    if len(docs) == 1:
        # Short enough — just summarize directly
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=MAP_PROMPT)
    else:
        # Long doc — use map-reduce
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            map_prompt=MAP_PROMPT,
            combine_prompt=COMBINE_PROMPT,
            verbose=False,
        )

    result = chain.invoke({"input_documents": docs})
    return result["output_text"]


if __name__ == "__main__":
    sample = """
    LangChain is an open-source framework developed to simplify the creation of applications
    using large language models. It was created by Harrison Chase and first released in October 2022.
    The framework provides a standard interface for chains, multiple integrations with other tools,
    and end-to-end chains for common applications.

    LangChain allows developers to create data-aware and agentic applications. Data-aware means
    connecting a language model to other data sources. Agentic means allowing a language model
    to interact with its environment. The framework includes components for models, prompts,
    indexes, memory, chains, and agents.

    As of 2024, LangChain has become one of the most popular frameworks for building LLM-powered
    applications, with thousands of GitHub stars and active community contributions. The ecosystem
    includes LangSmith for observability, LangServe for deployment, and LangGraph for stateful agents.
    """
    print(summarize(sample))
