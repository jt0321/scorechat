# AI Automation Portfolio

A modular AI automation framework built with LangChain, demonstrating multi-agent orchestration, tool use, and real-world workflow automation patterns.

## Architecture

```
ai-automation-portfolio/
├── agents/           # Specialized LangChain agents
│   ├── research_agent.py     # Web research + summarization agent
│   ├── data_agent.py         # Data analysis + SQL agent
│   └── orchestrator.py       # Multi-agent coordinator
├── chains/           # LangChain pipeline chains
│   ├── summarization_chain.py
│   ├── rag_chain.py          # Retrieval-Augmented Generation
│   └── extraction_chain.py   # Structured data extraction
├── tools/            # Custom LangChain tools
│   ├── sql_tool.py
│   ├── file_tool.py
│   └── web_tool.py
├── data/             # Sample data and vector store
└── logs/             # Run logs
```

## Features

- **Multi-Agent Orchestration** — Coordinator delegates tasks to specialized agents (research, data analysis, writing)
- **RAG Pipeline** — Retrieval-Augmented Generation over local documents using FAISS
- **SQL Agent** — Natural language to SQL query execution
- **Structured Extraction** — LLM-powered extraction from unstructured text into Pydantic models

## Stack

| Layer | Technology |
|-------|-----------|
| LLM Framework | LangChain |
| LLMs | OpenAI GPT-4o / Ollama (local) |
| Vector Store | FAISS |
| Data | SQLite / Pandas |
| Scheduling | APScheduler / cron |
| Language | Python 3.11+ |

## Setup

```bash
pip install -r requirements.txt

# Copy and fill in your keys
cp .env.example .env

# Run the orchestrator demo
python agents/orchestrator.py

```

## Agents

### Research Agent
Uses web search tools + LangChain to gather and summarize information on a topic. Outputs structured markdown reports.

### Data Agent
Translates natural language queries into SQL, executes them against a SQLite database, and explains results in plain English.

### Orchestrator
Routes tasks to the right agent based on task type using a ReAct-style loop. Maintains shared memory across agent calls.
