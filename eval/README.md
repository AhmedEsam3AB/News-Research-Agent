# Evaluation

Measures how often the agent produces a usable report, with and without the editor check
(the critique loop), on a fixed set of test cases.

```bash
python eval/run_eval.py
```

Needs the same `.env` as the app. It uses the free-model request quota: one full run is
roughly 150–250 requests.

## Test set

[`test_set.json`](test_set.json) has 10 cases, each a topic with a fixed past date window so
every run asks the same question:

- 7 companies a PR specialist might track: Boeing, Tesla, Nike, Starbucks, Ryanair, OpenAI, Nintendo
- 1 issue: EU AI Act enforcement
- 1 one-day window (Apple), a harder date constraint
- 1 made-up company (Bramblecote Organic Dairy). The right answer is "little or no coverage",
  with no invented stories.

## Modes

| Mode | What runs |
| --- | --- |
| `with_check` | The normal agent: draft, editor check, and up to 2 retries with the editor's feedback |
| `without_check` | The same agent with the editor check removed (`build_agent_graph(critique=False)`): the first draft is the report |

For each case the two modes run back to back. Changes in the news index during the run then
hit both modes equally.

## How a report is graded

A separate **judge model** grades every report. The judge comes from a different model family
than the agent, and it never sees which mode produced the report. It gets the topic, the date
window, the sources the agent collected (the same text the summary step saw) and the report.
It then answers five yes/no checks:

| Check | Passes when |
| --- | --- |
| `on_topic` | The report is about the requested topic. |
| `grounded` | Every specific claim (names, numbers, events, quotes) is supported by the sources. |
| `in_window` | Stories presented as news happened inside the date window. |
| `covers_main_stories` | The main stories in the sources are covered. |
| `neutral` | Neutral, factual tone. |

**A report passes only if the run finished and all five checks are yes.** For the made-up
company, it passes only if it clearly says little or no coverage was found and invents nothing.

The runner also records numbers that need no judge:
- sources collected
- the share of dated sources that fall inside the window
- source URLs cited in the report
- drafts, searches and articles read
- editor-check failures
- time taken

## Results

Each run writes to `results/<date>/`:
- **`runs.jsonl`:** one line per run, with the report, sources, numbers and the judge's
  verdict and reasons.
- **`summary.md`:** the pass rates and the table per case.

## Limits

- **One run per case per mode.** The agent is not deterministic (temperature 0.7) and live news
  search changes, so with 10 cases a difference of one or two passes is within noise.
- **The judge is an LLM.** Its verdicts are a consistent, blind second opinion, not ground
  truth. Its reasons are saved so any verdict can be checked by hand.
- **Free models only.** Results are for the models named in `summary.md`. Free models can be
  rate-limited mid-run; the runner waits and retries, and can resume a stopped run with `--out`.
