"""
Research Agent
--------------
Uses LangChain ReAct agent with web search + Wikipedia tools to
research a topic and return a structured markdown summary.
"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv()


def get_llm(temperature: float = 0.2):
    use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    if use_local:
        from langchain_community.llms import Ollama
        return Ollama(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def build_research_agent() -> AgentExecutor:
    llm = get_llm()

    tools = [
        DuckDuckGoSearchRun(name="web_search"),
        WikipediaQueryRun(
            name="wikipedia",
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=3000),
        ),
    ]

    template = """You are a thorough research assistant. Use your tools to answer the question with citations.

Tools available:
{tools}

Tool names: {tool_names}

Format your response as:
## Summary
<concise summary>

## Key Findings
- finding 1
- finding 2

## Sources
- source 1
- source 2

Use this format:
Thought: what should I do?
Action: tool_name
Action Input: query
Observation: result
... (repeat as needed)
Thought: I have enough info
Final Answer: <markdown formatted answer>

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)
    agent = create_react_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True,
    )


def research(topic: str) -> dict:
    """Research a topic and return structured results."""
    agent = build_research_agent()
    result = agent.invoke({"input": f"Research this topic thoroughly: {topic}"})

    output = {
        "topic": topic,
        "timestamp": datetime.utcnow().isoformat(),
        "result": result["output"],
        "agent": "research_agent",
    }

    # Log run
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/research_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    result = research("LangChain multi-agent systems in production")
    print(result["result"])
