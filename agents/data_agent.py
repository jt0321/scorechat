"""
Data Agent
----------
Natural language → SQL agent. Translates plain English questions
into SQL queries, runs them against a SQLite database, and explains
results in plain English.
"""

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv
import os
import sqlite3
import pandas as pd
from datetime import datetime
import json

load_dotenv()


def seed_sample_db(db_path: str = "data/sample.db"):
    """Create and seed a sample SQLite database for demos."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            diagnosis TEXT,
            admission_date TEXT,
            discharge_date TEXT,
            cost REAL
        );

        CREATE TABLE IF NOT EXISTS pipeline_runs (
            id INTEGER PRIMARY KEY,
            run_date TEXT,
            records_processed INTEGER,
            errors INTEGER,
            duration_seconds REAL,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            run_date TEXT,
            accuracy REAL,
            f1_score REAL,
            latency_ms REAL
        );
    """)

    # Seed if empty
    if cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0:
        patients = [
            ("Alice M.", 34, "Type 2 Diabetes", "2024-01-10", "2024-01-15", 4200.50),
            ("Bob K.", 58, "Hypertension", "2024-02-01", "2024-02-04", 2100.00),
            ("Carol T.", 45, "Asthma", "2024-02-20", "2024-02-22", 1500.75),
            ("David R.", 72, "Heart Failure", "2024-03-05", "2024-03-20", 18500.00),
            ("Eve S.", 29, "Appendicitis", "2024-03-11", "2024-03-14", 9800.00),
        ]
        cur.executemany(
            "INSERT INTO patients (name, age, diagnosis, admission_date, discharge_date, cost) VALUES (?,?,?,?,?,?)",
            patients,
        )

    if cur.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 0:
        runs = [
            ("2024-04-01", 12000, 3, 45.2, "success"),
            ("2024-04-02", 11800, 0, 42.1, "success"),
            ("2024-04-03", 0, 1, 2.5, "failed"),
            ("2024-04-04", 13200, 1, 48.9, "success"),
        ]
        cur.executemany(
            "INSERT INTO pipeline_runs (run_date, records_processed, errors, duration_seconds, status) VALUES (?,?,?,?,?)",
            runs,
        )

    conn.commit()
    conn.close()
    return db_path


def build_data_agent(db_path: str = "data/sample.db") -> object:
    seed_sample_db(db_path)
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )
    return agent


def query(question: str, db_path: str = "data/sample.db") -> dict:
    agent = build_data_agent(db_path)
    result = agent.invoke({"input": question})

    output = {
        "question": question,
        "timestamp": datetime.utcnow().isoformat(),
        "answer": result["output"],
        "agent": "data_agent",
    }

    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/data_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    result = query("Which diagnosis had the highest average treatment cost?")
    print(result["answer"])
