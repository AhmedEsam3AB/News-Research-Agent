# News Research Agent: Case Study

## Problem

PR specialists need to know what the press has said about a client, a competitor or an issue over a set period, so they can brief the client and decide how to respond. Gathering that by hand means running many searches, checking each story falls inside the right dates, and condensing it all into a neutral summary, which is slow, easy to get wrong, and has to be redone every time the topic or period changes.

## What the system does

It is one LangGraph workflow (`StateGraph` in `agent.py`) with five nodes that share a single state object. GPT-4o writes the queries and the summary, and separate GPT-4o calls in `evaluator.py` check the inputs and the output. It is a single pipeline with a retry loop, not a set of separate agents.

Step names follow the README's Flow section; the code name is in brackets so each step can be traced.

- **Topic Input** (`plan_node` + `evaluate_topic`). Asks for a topic. An LLM fixes typos and rejects topics that are gibberish or too vague. It keeps asking until the topic passes.
- **Time Input** (`plan_node` + `evaluate_time_period`). Asks for a time period. It accepts phrases like "last week" or "past 3 days", fixes typos, and rejects future dates. It keeps asking until the period passes.
- **Query Generation** (`research_plan_node`). GPT-4o turns the topic and period into 3 search queries and uses today's date to turn relative dates into YYYY-MM-DD. If this fails, it falls back to one query: `"{topic} {time_period} news"`. A check (`evaluate_queries`) prints a warning if the queries look off-topic or too alike, but the run carries on anyway.
- **Search** (`generate_node`). Runs each query through Tavily, keeps the top 3 results per query, and collects each result's URL and text. A query that errors is skipped. If nothing comes back, the next step receives "No results found."
- **Summarize** (`create_node`). GPT-4o writes an executive summary of the search results for the topic.
- **Critique** (`search_critique_node` + `evaluate_critique`, routed by `check_satisfaction`). A "senior editor" check asks whether the summary is comprehensive, neutral, and on-topic. If it fails, the feedback goes back to Query Generation to write new queries. After 2 retries (3 drafts in total) it accepts the latest draft even if it still failed. The README has no name for this step, so "Critique" is mine.
- **Report** (the `__main__` block in `agent.py`). Prints the summary to the terminal.

## Open questions

1. **`llm.py` doesn't exist.** It isn't in the repo or its git history. The model setup lives inline instead: `agent.py:36` (GPT-4o, temperature 0.7) and `evaluator.py:9` (GPT-4o, temperature 0). Is there another branch or file I should be reading?
2. **The time period isn't enforced at search time.** Tavily gets no date filter; the dates exist only inside the query text. Summarize and Critique never see the time period, so stories from outside the window can reach the summary unchecked. For PR work, where the date window is the point, should the case study describe this as a known limitation?
3. **A failed draft can still become the final report.** When retries run out, the last draft is printed with no flag in the report itself. The only sign is a console line printed during the run. Should the case study call this out?
4. **Sources may not appear in the summary.** URLs are passed to Summarize, but its prompt doesn't ask for citations, so whether the report cites sources depends on the model. A PR specialist would usually need to show where coverage appeared. Should we claim the output is sourced?
5. **The query check only sees part of its input.** `evaluate_queries` says it checks the queries against the "topic and time period", but only the topic is passed in. Should this be noted, fixed later, or left out?
6. **Should we keep calling it an "agent"?** The README does. The step order is fixed. The model never chooses which tool to call next. The only decision it makes is pass or retry at the critique step. "Research workflow with a self-critique loop" is the more accurate label.
7. **Numbers to measure later.** Nothing has been measured yet, including how often the critique fails ([X]%), average drafts per run ([X]), and time per run ([X] s).
