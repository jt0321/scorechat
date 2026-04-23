"""
Multi-Agent Orchestrator
------------------------
Routes incoming tasks to the appropriate specialized agent using
a lightweight classification step, then returns the final result.

Agents:
  - research_agent  → web search, summarization, fact-finding
  - data_agent      → SQL, data analysis, metrics questions
  - default         → direct LLM response for general tasks
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()

# Lazy imports to avoid loading unused agents
def _get_research_agent():
    from agents.research_agent import research
    return research

def _get_data_agent():
    from agents.data_agent import query
    return query


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a task router. Classify the user's task into exactly one category:

- "research"  → needs web search, summaries, current events, explanations
- "data"      → needs SQL queries, database analysis, metrics, statistics
- "general"   → everything else (writing, code, advice, planning)

Respond with only the category word. No explanation."""),
    ("human", "{task}"),
])


def classify_task(task: str, llm) -> str:
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    label = chain.invoke({"task": task}).strip().lower()
    if label not in ("research", "data", "general"):
        label = "general"
    return label


def run(task: str) -> dict:
    """Main entry point. Pass any task string, get back a result dict."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent_type = classify_task(task, llm)
    print(f"[orchestrator] Routing to: {agent_type}")

    if agent_type == "research":
        result = _get_research_agent()(task)
        answer = result["result"]

    elif agent_type == "data":
        result = _get_data_agent()(task)
        answer = result["answer"]

    else:
        # Direct LLM for general tasks
        chain = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a helpful AI assistant. Be concise and direct."),
                ("human", "{task}"),
            ])
            | llm
            | StrOutputParser()
        )
        answer = chain.invoke({"task": task})

    output = {
        "task": task,
        "agent": agent_type,
        "answer": answer,
        "timestamp": datetime.utcnow().isoformat(),
    }

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/orchestrator_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TASK:   {task}")
    print(f"AGENT:  {agent_type}")
    print(f"ANSWER:\n{answer}")
    print(f"{'='*60}\n")

    return output


if __name__ == "__main__":
    # Demo: run a few tasks through the orchestrator
    tasks = [
        "What are the latest trends in AI agent frameworks?",
        "Which pipeline run had the most errors in the database?",
        "Write a one-paragraph bio for a data engineer who specializes in AI automation.",
    ]
    for t in tasks:
        run(t)
        print()
