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

## Flow
1. **Topic Input**: Asks for a topic. Evaluator checks clarity.
2. **Time Input**: Asks for a time period. Evaluator checks validity.
3. **Query Generation**: Plans search queries. Evaluator checks relevance.
4. **Search**: Uses Tavily to find news.
5. **Summarize**: Synthesizes findings.
6. **Report**: Prints the final report.
