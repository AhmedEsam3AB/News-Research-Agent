import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from llm import get_llm, with_schema
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
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
    critique_passed: bool  # False when the draft was accepted only because retries ran out
    loop_count: int

    # Research agent (agent mode only)
    messages: Annotated[list, add_messages]
    tool_rounds: int

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
        return {"is_satisfactory": True, "critique_passed": True, "critique_feedback": ""}
    else:
        print(f"[!] Critique Failed: {critique.feedback}")

        # Check Max Loops
        if loop_count >= 2:
            print("[!] Max retries reached. Accepting best effort.")
            return {"is_satisfactory": True, "critique_passed": False, "critique_feedback": critique.feedback}

        return {
            "is_satisfactory": False,
            "critique_passed": False,
            "critique_feedback": critique.feedback,
            "loop_count": loop_count + 1
        }

# --- Research Agent (agent mode) ---
# Replaces RESEARCH_PLAN + GENERATE with a tool-calling loop: the model decides
# what to search, whether to read an article in full, and when it has enough.

MAX_TOOL_ROUNDS = 6        # tool-calling turns allowed per research round
ARTICLE_CHAR_LIMIT = 4000  # cap on text returned by read_article
TOOL_MISSES = ("Search failed", "Could not read", "No results found")

_tavily = None

def _tavily_client():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient()
    return _tavily

@tool(parse_docstring=True)
def search_news(query: str, start_date: str, end_date: str) -> str:
    """Search news articles published between start_date and end_date.

    Args:
        query: What to search for.
        start_date: Earliest publish date, in YYYY-MM-DD format.
        end_date: Latest publish date, in YYYY-MM-DD format.
    """
    try:
        res = _tavily_client().search(
            query, topic="news", start_date=start_date, end_date=end_date, max_results=5
        )
    except Exception as e:
        return f"Search failed for '{query}': {e}"
    items = res.get("results", [])
    if not items:
        return f"No results found for '{query}'."
    return "\n---\n".join(
        f"Source: {i.get('url')}\nPublished: {i.get('published_date', 'unknown')}\n"
        f"Title: {i.get('title')}\nContent: {i.get('content')}"
        for i in items
    )

@tool(parse_docstring=True)
def read_article(url: str) -> str:
    """Fetch the full text of an article when its search snippet is not enough.

    Args:
        url: The article URL, taken from a search_news result.
    """
    try:
        res = _tavily_client().extract(urls=[url])
    except Exception as e:
        return f"Could not read {url}: {e}"
    items = res.get("results", [])
    if not items or not items[0].get("raw_content"):
        return f"Could not read {url}."
    return f"Source: {url}\nContent: {items[0]['raw_content'][:ARTICLE_CHAR_LIMIT]}"

RESEARCH_TOOLS = [search_news, read_article]
research_llm = llm.bind_tools(RESEARCH_TOOLS)
# Used once the step limit is hit: same tools visible, but the model may not call them.
wrap_up_llm = llm.bind_tools(RESEARCH_TOOLS, tool_choice="none")

def research_brief_node(state: AgentState):
    """
    RESEARCH_BRIEF: Gives the agent its task, or the editor's feedback on a retry.
    """
    print("\n--- Step: RESEARCH BRIEF ---")
    if state.get("messages"):
        feedback = state.get("critique_feedback", "")
        print(f"[*] Passing critique feedback to the agent: {feedback}")
        messages = [HumanMessage(content=(
            f"An editor rejected the summary built from your research: {feedback}\n"
            "Research further to fix this, then stop."
        ))]
    else:
        current_date = datetime.now().strftime("%Y-%m-%d")
        messages = [
            SystemMessage(content=(
                f"You are a news research agent. Today's date is {current_date}.\n"
                "Gather news coverage of the user's topic within their time period, using your tools.\n"
                "- Convert the time period into start_date and end_date (YYYY-MM-DD) and pass them to every search_news call.\n"
                "- Search from several angles. Look at what each search returns before deciding what to search next.\n"
                "- Use read_article when a snippet is too thin to understand the story.\n"
                "- Stop calling tools once you have enough coverage of the period, then reply with a short list of the main stories you found."
            )),
            HumanMessage(content=f"Topic: {state['topic']}\nTime period: {state['time_period']}"),
        ]
    return {"messages": messages, "tool_rounds": 0}

def research_agent_node(state: AgentState):
    """
    RESEARCH_AGENT: The model decides whether to search, read an article, or stop.
    """
    rounds = state.get("tool_rounds", 0)
    at_limit = rounds >= MAX_TOOL_ROUNDS
    if at_limit:
        print(f"[!] Step limit ({MAX_TOOL_ROUNDS}) reached. Telling the agent to stop.")
    try:
        response = (wrap_up_llm if at_limit else research_llm).invoke(state["messages"])
    except Exception as e:
        # Many OpenRouter models reject requests that include tools.
        raise RuntimeError(
            f"Agent mode needs a model that supports tool calling. "
            f"The call to {getattr(llm, 'model_name', 'the model')} failed: {e}\n"
            "Set OPENROUTER_MODEL to a model with tool support, or run with --baseline."
        ) from e

    if response.tool_calls:
        for call in response.tool_calls:
            print(f"[*] Agent calls {call['name']}({call['args']})")
    elif not at_limit:
        print("[*] Agent decided it has enough.")
    return {"messages": [response], "tool_rounds": rounds + 1}

def collect_node(state: AgentState):
    """
    COLLECT: Hands everything the agent's tools returned to CREATE.
    """
    found = [
        m.content for m in state["messages"]
        if isinstance(m, ToolMessage) and not m.content.startswith(TOOL_MISSES)
    ]
    print(f"[*] Collected {len(found)} tool results for the summary.")
    return {"search_results": "\n---\n".join(found) or "No results found."}

# --- Conditional Edges ---

def check_satisfaction(state: AgentState):
    if state["is_satisfactory"]:
        return END
    return "research_plan" # Loop back to query generation

def check_satisfaction_agent(state: AgentState):
    if state["is_satisfactory"]:
        return END
    return "research_brief" # Loop back to the agent with the critique feedback

def route_research(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return "collect"

def route_followup(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

# --- Follow-up chat (used by the GUI) ---
# After a report, the PR specialist can ask questions. The same agent node and
# tools answer them: the agent uses the report first and searches again only if needed.

FOLLOWUP_SOURCES_LIMIT = 15000  # characters of collected sources given to the follow-up agent

def followup_system_message(topic: str, time_period: str, summary: str, search_results: str):
    current_date = datetime.now().strftime("%Y-%m-%d")
    return SystemMessage(content=(
        f"You are a news research agent helping a PR specialist. Today's date is {current_date}.\n"
        f"You researched '{topic}' for the period {time_period} and wrote the report below.\n"
        "Answer the specialist's follow-up questions. Use the report and sources first. "
        "Call search_news or read_article only when they don't cover the question; "
        "search the same period unless the question asks for another. "
        "Keep answers short and cite source URLs.\n\n"
        f"REPORT:\n{summary}\n\nSOURCES:\n{search_results[:FOLLOWUP_SOURCES_LIMIT]}"
    ))

# --- Graph Construction ---

def build_workflow_graph():
    """Baseline: the original fixed workflow (always 3 queries, top 3 results each)."""
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

    return workflow.compile()

def build_agent_graph(interactive: bool = True):
    """Agent: the model chooses its own searches and decides when to stop.

    interactive=False skips PLAN (terminal input). The caller must then pass
    topic, time_period and loop_count=0 in the initial state, as the GUI does.
    """
    graph = StateGraph(AgentState)

    if interactive:
        graph.add_node("plan", plan_node)
    graph.add_node("research_brief", research_brief_node)
    graph.add_node("research_agent", research_agent_node)
    graph.add_node("tools", ToolNode(RESEARCH_TOOLS))
    graph.add_node("collect", collect_node)
    graph.add_node("create", create_node)
    graph.add_node("search_critique", search_critique_node)

    if interactive:
        graph.set_entry_point("plan")
        graph.add_edge("plan", "research_brief")
    else:
        graph.set_entry_point("research_brief")
    graph.add_edge("research_brief", "research_agent")
    graph.add_conditional_edges("research_agent", route_research, ["tools", "collect"])
    graph.add_edge("tools", "research_agent")
    graph.add_edge("collect", "create")
    graph.add_edge("create", "search_critique")
    graph.add_conditional_edges("search_critique", check_satisfaction_agent, ["research_brief", END])

    return graph.compile()

def build_followup_graph():
    """Follow-up chat: the agent answers one question, calling tools only if it needs to."""
    graph = StateGraph(AgentState)

    graph.add_node("research_agent", research_agent_node)
    graph.add_node("tools", ToolNode(RESEARCH_TOOLS))

    graph.set_entry_point("research_agent")
    graph.add_conditional_edges("research_agent", route_followup, ["tools", END])
    graph.add_edge("tools", "research_agent")

    return graph.compile()

# Worst case per research round: brief + 7 agent turns + 6 tool runs + collect + create + critique = 17 steps.
# Three rounds (first try + 2 retries) plus PLAN is 52, so 100 leaves headroom.
RECURSION_LIMIT = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research a news topic and print an executive summary.")
    parser.add_argument("--baseline", action="store_true",
                        help="Run the original fixed workflow instead of the research agent.")
    args = parser.parse_args()

    if args.baseline:
        print("Starting News Research (baseline fixed workflow)...")
        app = build_workflow_graph()
    else:
        print("Starting News Research Agent...")
        app = build_agent_graph()
    result = app.invoke({}, {"recursion_limit": RECURSION_LIMIT})

    print("\n" + "="*40)
    print("       FINAL RESEARCH REPORT       ")
    print("="*40)
    print(result["summary"])
    print("="*40 + "\n")
