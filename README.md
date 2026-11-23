# News Research Agent

A LangChain-based agent that researches news topics using Tavily and LangGraph.

## Features
- **LangGraph Workflow**: Structured flow with loops for validation.
- **Self-Correction**: Evaluates user input (topic, time) and generated queries using an LLM evaluator.
- **Tavily Search**: Retrieves realtime news.
- **Summarization**: Generates an executive summary of findings.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   Copy `.env.example` to `.env` and fill in your API keys.
   ```bash
   cp .env.example .env
   ```
   You need:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`

## Usage

Run the agent:
```bash
python agent.py
```

Run the original fixed workflow instead (3 queries, top 3 results each), kept as a baseline for comparison:
```bash
python agent.py --baseline
```

In agent mode, Query Generation and Search are replaced by a research agent. It has two tools: `search_news` (Tavily news search with a date range) and `read_article` (full article text). It decides what to search and when it has enough, up to 6 tool turns per round. Critique feedback goes back to the agent.

## Flow
1. **Topic Input**: Asks for a topic. Evaluator checks clarity.
2. **Time Input**: Asks for a time period. Evaluator checks validity.
3. **Query Generation**: Plans search queries. Evaluator checks relevance.
4. **Search**: Uses Tavily to find news.
5. **Summarize**: Synthesizes findings.
6. **Report**: Prints the final report.
