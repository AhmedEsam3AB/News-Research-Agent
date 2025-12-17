from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

llm = ChatOpenAI(temperature=0, model="gpt-4o")

from typing import Optional

class EvaluationResult(BaseModel):
    is_valid: bool = Field(description="Whether the input is valid and clear (after correction if possible)")
    feedback: str = Field(description="Feedback on what is wrong or needs clarification")
    corrected_value: Optional[str] = Field(description="The corrected input text if there were typos or minor issues. If no correction needed, this should be the original input.")

def evaluate_topic(topic: str) -> EvaluationResult:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator for a news research agent. Check if the user's topic is clear and specific. If there are typos, correct them. If the topic is gibberish or too vague, mark as invalid."),
        ("user", "Topic: {topic}")
    ])
    chain = prompt | llm.with_structured_output(EvaluationResult)
    res = chain.invoke({"topic": topic})
    # Fallback if LLM doesn't populate corrected_value
    if res.corrected_value is None:
        res.corrected_value = topic
    return res

def evaluate_time_period(time_period: str) -> EvaluationResult:
    current_date = datetime.now().strftime("%Y-%m-%d")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are an evaluator. Current Date: {current_date}. Check if the time period is a valid duration, date range, or natural language reference relative to the current date (e.g., 'today', 'yesterday', 'last month', 'this week', 'past 3 days'). Fix any typos (e.g., 'lst week' -> 'last week'). Future dates are invalid."),
        ("user", "Time Period: {time_period}")
    ])
    chain = prompt | llm.with_structured_output(EvaluationResult)
    res = chain.invoke({"time_period": time_period})
    if res.corrected_value is None:
        res.corrected_value = time_period
    return res

def evaluate_queries(original_topic: str, queries: list[str]) -> EvaluationResult:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an evaluator. Check if the generated search queries are relevant to the topic and time period, and diverse enough."),
        ("user", "Topic: {topic}\nQueries: {queries}")
    ])
    chain = prompt | llm.with_structured_output(EvaluationResult)
    return chain.invoke({"topic": original_topic, "queries": str(queries)})

def evaluate_critique(topic: str, summary: str) -> EvaluationResult:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior editor. Critique the following news summary. Is it comprehensive, neutral, and relevant to the topic? If it says 'No results found', mark as invalid. Return feedback for improvement."),
        ("user", "Topic: {topic}\nSummary: {summary}")
    ])
    chain = prompt | llm.with_structured_output(EvaluationResult)
    return chain.invoke({"topic": topic, "summary": summary})
