import os
import sys
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from llm import get_llm, with_schema
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient

from datetime import datetime
import operator

# Import evaluators
from evaluator import evaluate_topic, evaluate_time_period, evaluate_queries, evaluate_critique

# State Definition
class AgentState(TypedDict):
    # Inputs
    topic: str
    time_period: str
    
    # Internal Logic
    search_queries: List[str]
    search_results: str
    summary: str
    
    # Validation/Critique
    critique_feedback: str
    is_satisfactory: bool
    loop_count: int

# LLM Setup
llm = get_llm(temperature=0.7)

# --- Nodes ---

def plan_node(state: AgentState):
    """
    PLAN: Handles user input and validation for Topic and Time.
    Combines the interactivity of previous nodes into one robust planning step.
    """
    print("\n--- Step: PLAN ---")
    topic = state.get("topic", "")
    time_period = state.get("time_period", "")
    
    # 1. Get/Validate Topic
    while True:
        if not topic:
             topic = input("\n> Please enter the news topic you are interested in: ")
        
        t_res = evaluate_topic(topic)
        final_topic = t_res.corrected_value if t_res.corrected_value else topic
        
        if final_topic != topic:
            print(f"[*] Auto-corrected topic to: {final_topic}")
            topic = final_topic
            
        if t_res.is_valid:
            break
        else:
            print(f"[!] Topic issue: {t_res.feedback}")
            topic = "" # Reset to ask again
            
    # 2. Get/Validate Time
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_month = datetime.now().strftime("%B %Y")
    
    while True:
        if not time_period:
            time_period = input(f"\n> Please enter the time period (e.g., 'last 7 days', '{current_month}'): ")
            
        tp_res = evaluate_time_period(time_period)
        final_time = tp_res.corrected_value if tp_res.corrected_value else time_period
        
        if final_time != time_period:
             print(f"[*] Auto-corrected time to: {final_time}")
             time_period = final_time
             
        if tp_res.is_valid:
            break
        else:
             print(f"[!] Time issue: {tp_res.feedback}")
             time_period = ""

    return {"topic": topic, "time_period": time_period, "loop_count": 0}

def research_plan_node(state: AgentState):
    """
    RESEARCH_PLAN: Generates search queries.
    Utilizes feedback if this is a retry.
    """
    print("\n--- Step: RESEARCH PLAN (Query Generation) ---")
    topic = state["topic"]
    time_period = state["time_period"]
    feedback = state.get("critique_feedback", "")
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    prompt_text = f"Current Date: {current_date}\nGenerate 3 diverse google search queries to find news about '{topic}' from '{time_period}'."
    if feedback:
        print(f"[*] Incorporating critique feedback: {feedback}")
        prompt_text += f"\n\nCRITICAL FEEDBACK FROM PREVIOUS ATTEMPT: {feedback}\nAdjust queries to address this feedback."
        
    messages = [
        ("system", "You are a search query expert. Use the Current Date to convert relative time expressions (today, yesterday, last week) into specific YYYY-MM-DD format absolute dates for Tavily. Ensure queries are targeted."), 
        ("user", prompt_text)
    ]
    
    # Use JSON mode for robustness
    class Queries(TypedDict):
        queries: List[str]
        
    query_gen = with_schema(llm, Queries)
    try:
        result = query_gen.invoke(messages)
        queries = result["queries"]
    except Exception as e:
        # Fallback
        print(f"[!] Query generation failed ({e}); using a basic query.")
        queries = [f"{topic} {time_period} news"]
        
    # Validation step (internal)
    val_res = evaluate_queries(topic, queries)
    if not val_res.is_valid:
        print(f"[!] Query validation warning: {val_res.feedback}. Proceeding anyway to explore.")
        
    return {"search_queries": queries}

def generate_node(state: AgentState):
    """
    GENERATE: Executes the search (creating knowledge).
    """
    print("\n--- Step: GENERATE (Execution) ---")
    queries = state["search_queries"]
    print(f"[*] Searching Tavily for: {queries}")
    
    client = TavilyClient()
    all_results = []

    for q in queries:
        try:
            res = client.search(query=q, max_results=3)
            # Tavily returns {"results": [{url, content, ...}, ...]}
            for item in res.get("results", []):
                all_results.append(f"Source: {item.get('url')}\nContent: {item.get('content')}\n")
        except Exception as e:
            print(f"[!] Error searching {q}: {e}")
            
    combined_results = "\n---\n".join(all_results)
    if not combined_results.strip():
        combined_results = "No results found."
        
    return {"search_results": combined_results}

def create_node(state: AgentState):
    """
    CREATE: Synthesizes the report from the search results.
    """
    print("\n--- Step: CREATE (Drafting Report) ---")
    results = state["search_results"]
    topic = state["topic"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional news analyst. Create an executive summary of the following search results."),
        ("user", f"Topic: {topic}\n\nSearch Results:\n{results}")
    ])
    chain = prompt | llm
    res = chain.invoke({})
    summary = res.content
    
    # Print primarily for user visibility, though 'present' isn't explicitly in the user's requested list,
    # usually 'create' produces the artifact.
    # We will print it at the end if valid.
    return {"summary": summary}

def search_critique_node(state: AgentState):
    """
    SEARCH_CRITIQUE: Evaluates the quality of the report.
    Decides if we need to loop back.
    """
    print("\n--- Step: SEARCH CRITIQUE ---")
    topic = state["topic"]
    summary = state["summary"]
    loop_count = state.get("loop_count", 0)
    
    # 1. Evaluate
    critique = evaluate_critique(topic, summary)
    
    if critique.is_valid:
        print("[*] Critique Passed: Report is satisfactory.")
        return {"is_satisfactory": True, "critique_feedback": ""}
    else:
        print(f"[!] Critique Failed: {critique.feedback}")
        
        # Check Max Loops
        if loop_count >= 2:
            print("[!] Max retries reached. Accepting best effort.")
            return {"is_satisfactory": True, "critique_feedback": ""}
            
        return {
            "is_satisfactory": False, 
            "critique_feedback": critique.feedback,
            "loop_count": loop_count + 1
        }

# --- Conditional Edges ---

def check_satisfaction(state: AgentState):
    if state["is_satisfactory"]:
        return END
    return "research_plan" # Loop back to query generation

# --- Graph Construction ---

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("plan", plan_node)
workflow.add_node("research_plan", research_plan_node)
workflow.add_node("generate", generate_node)
workflow.add_node("create", create_node)
workflow.add_node("search_critique", search_critique_node)

# Set Entry
workflow.set_entry_point("plan")

# Add Edges
workflow.add_edge("plan", "research_plan")
workflow.add_edge("research_plan", "generate")
workflow.add_edge("generate", "create")
workflow.add_edge("create", "search_critique")

# Conditional Edge from Critique
workflow.add_conditional_edges("search_critique", check_satisfaction)

app = workflow.compile()

if __name__ == "__main__":
    print("Starting News Research Agent (Refactored Flow)...")
    result = app.invoke({})
    
    print("\n" + "="*40)
    print("       FINAL RESEARCH REPORT       ")
    print("="*40)
    print(result["summary"])
    print("="*40 + "\n")
