"""
Structured Extraction Chain
----------------------------
Uses LangChain + Pydantic to extract typed structured data
from unstructured text — job postings, reports, emails, etc.
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
import os
import json

load_dotenv()


# ── Schema definitions ──────────────────────────────────────────────────────

class JobPosting(BaseModel):
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location or 'Remote'")
    salary_range: Optional[str] = Field(None, description="Salary range if mentioned")
    required_skills: List[str] = Field(description="List of required technical skills")
    experience_years: Optional[int] = Field(None, description="Years of experience required")
    is_remote: bool = Field(description="Whether the role is remote")


class PipelineRunSummary(BaseModel):
    pipeline_name: str = Field(description="Name of the pipeline")
    status: str = Field(description="Run status: success, failed, or partial")
    records_processed: Optional[int] = Field(None, description="Number of records processed")
    errors: List[str] = Field(default_factory=list, description="List of error messages")
    duration_seconds: Optional[float] = Field(None, description="Run duration in seconds")


# ── Chain builders ───────────────────────────────────────────────────────────

def build_extraction_chain(schema: type):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    parser = PydanticOutputParser(pydantic_object=schema)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract structured information from the text. {format_instructions}"),
        ("human", "{text}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


# ── Helper functions ─────────────────────────────────────────────────────────

def extract_job_posting(text: str) -> JobPosting:
    chain = build_extraction_chain(JobPosting)
    return chain.invoke({"text": text})


def extract_pipeline_summary(text: str) -> PipelineRunSummary:
    chain = build_extraction_chain(PipelineRunSummary)
    return chain.invoke({"text": text})


if __name__ == "__main__":
    sample_job = """
    We're hiring a Senior Data Engineer at Acme Corp (Remote).
    Salary: $140k–$170k. Must have 5+ years of experience with Python,
    Apache Spark, dbt, and GCP. Experience with Kafka is a plus.
    """
    job = extract_job_posting(sample_job)
    print("Job Extraction:")
    print(json.dumps(job.model_dump(), indent=2))

    sample_run = """
    Healthcare ingestion pipeline completed at 03:42 UTC.
    Processed 14,892 patient records in 47.3 seconds.
    Encountered 2 errors: 'null SSN on row 4021' and 'date parse failure on row 9103'.
    Overall status: partial success.
    """
    run = extract_pipeline_summary(sample_run)
    print("\nPipeline Run Extraction:")
    print(json.dumps(run.model_dump(), indent=2))
