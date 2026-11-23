# News Research Agent: Case Study

## Problem

People who need a quick, reliable picture of what happened on a news topic during a specific window, such as an analyst or someone preparing a briefing, have to come up with search terms, run several searches, and condense the results into a short summary. Doing this by hand is slow and uneven: weak search terms miss coverage, dates are easy to get wrong, and nobody checks whether the summary is complete and neutral before it gets used.

## What the system does

It is one LangGraph workflow (`StateGraph` in `agent.py`) with five nodes that share a single state object. GPT-4o writes the queries and the summary, and separate GPT-4o calls in `evaluator.py` check the inputs and the output. It is a single pipeline with a retry loop, not a set of separate agents.

- **Asks for a topic and checks it** (`plan_node` + `evaluate_topic`). An LLM fixes typos and rejects topics that are gibberish or too vague. It keeps asking until the topic passes.
- **Asks for a time period and checks it** (`plan_node` + `evaluate_time_period`). It accepts phrases like "last week" or "past 3 days", fixes typos, and rejects future dates. It keeps asking until the period passes.
- **Writes three search queries** (`research_plan_node`). GPT-4o turns the topic and period into 3 queries and uses today's date to turn relative dates into YYYY-MM-DD. If query generation fails, it falls back to one query: `"{topic} {time_period} news"`.
- **Checks the queries, but only warns** (`evaluate_queries`, called inside `research_plan_node`). If the queries look off-topic or too alike, it prints a warning and carries on anyway.
- **Searches the web** (`generate_node`). It runs each query through Tavily, keeps the top 3 results per query, and collects each result's URL and text. A query that errors is skipped. If nothing comes back, the next step receives "No results found."
- **Drafts an executive summary** (`create_node`). GPT-4o summarises the collected search results for the topic.
- **Critiques the draft and retries if it falls short** (`search_critique_node` + `evaluate_critique`, routed by `check_satisfaction`). A "senior editor" check asks whether the summary is comprehensive, neutral, and on-topic. If it fails, the feedback goes back to `research_plan_node` to write new queries. After 2 retries (3 drafts in total) it accepts the latest draft even if it still failed.
- **Prints the final report** (the `__main__` block in `agent.py`). The summary is printed to the terminal.

## Open questions

1. **`llm.py` doesn't exist.** It isn't in the repo or its git history. The model setup lives inline instead: `agent.py:36` (GPT-4o, temperature 0.7) and `evaluator.py:9` (GPT-4o, temperature 0). Is there another branch or file I should be reading?
2. **Who is the user?** The "analyst or someone preparing a briefing" in the Problem section is my guess. Pick the audience you want the case study to speak to.
3. **The time period isn't enforced at search time.** Tavily gets no date filter; the dates exist only inside the query text. `create_node` and `evaluate_critique` never see the time period, so results from outside the window can reach the summary unchecked. Should the case study describe this as a known limitation?
4. **A failed draft can still become the final report.** When retries run out, the last draft is printed with no flag in the report itself. The only sign is a console line printed during the run. Should the case study call this out?
5. **Sources may not appear in the summary.** URLs are passed to `create_node`, but its prompt doesn't ask for citations, so whether the report cites sources depends on the model. Should we claim the output is sourced?
6. **The query check only sees part of its input.** `evaluate_queries` says it checks the queries against the "topic and time period", but only the topic is passed in. Should this be noted, fixed later, or left out?
7. **Which step names should the case study use?** The code calls the search step `generate` and the summary step `create`. The README calls them "Search" and "Summarize". I used the code names so they can be traced.
8. **Should we keep calling it an "agent"?** The README does. The step order is fixed. The model never chooses which tool to call next. The only decision it makes is pass or retry at the critique step. "Research pipeline with a self-critique loop" may be the more accurate label.
9. **Numbers to measure later.** Nothing has been measured yet, including how often the critique fails ([X]%), average drafts per run ([X]), and time per run ([X] s).
