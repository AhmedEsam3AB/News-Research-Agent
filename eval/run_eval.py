"""Run the fixed test set with and without the editor check, then grade every report.

Usage (from the repo root):
    python eval/run_eval.py                       # all cases, both modes
    python eval/run_eval.py --cases boeing,tesla  # a subset
    python eval/run_eval.py --out eval/results/2026-11-25   # resume an earlier run

Modes:
    with_check     the normal agent: draft -> editor check -> up to 2 retries
    without_check  the same agent with the editor check removed: first draft is the report

Each report is graded by a separate judge model that never sees which mode produced it.
A report PASSES when the run finished and the judge says yes to all five checks:
on_topic, grounded, in_window, covers_main_stories, neutral.

Results go to <out>/runs.jsonl (one line per run, written as it goes, so a stopped run
can be resumed) and <out>/summary.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

import agent  # noqa: E402
from llm import get_llm, with_schema  # noqa: E402

MODES = ["with_check", "without_check"]
CHECKS = ["on_topic", "grounded", "in_window", "covers_main_stories", "neutral"]
DEFAULT_JUDGE = "baidu/ernie-5.5-flash-preview:free"  # different model family from the agent


# --- Judge ----------------------------------------------------------------------

class Verdict(BaseModel):
    on_topic: bool = Field(description="The report is about the requested topic.")
    grounded: bool = Field(description="Every specific claim (names, numbers, events) is supported by the listed sources.")
    in_window: bool = Field(description="The stories presented as news happened inside the date window.")
    covers_main_stories: bool = Field(description="The main stories visible in the sources are covered; nothing major is left out.")
    neutral: bool = Field(description="Neutral, factual tone; not promotional or one-sided.")
    reason: str = Field(description="One or two sentences explaining any check that failed, or 'All checks pass.'")


JUDGE_SYSTEM = (
    "You grade news coverage reports written for a PR specialist. You get the topic, the date window, "
    "the sources the researcher found (title, URL, publish date, snippet) and the report. Judge only "
    "against these sources, not your own knowledge. Be strict: a check is true only if it clearly holds.\n"
    "- on_topic: the report is about the requested topic, not a different company or subject.\n"
    "- grounded: every specific claim (names, numbers, events, quotes) is supported by the sources. "
    "General framing is fine. Any claim not in the sources makes this false.\n"
    "- in_window: the stories presented as news happened inside the date window, judging by source dates "
    "and dates in the report. Background clearly labelled as background is fine.\n"
    "- covers_main_stories: the report covers the main stories the sources show for this topic and window.\n"
    "- neutral: neutral, factual tone; not promotional or one-sided."
)

NO_COVERAGE_NOTE = (
    "\nThis case uses a made-up company, so there should be little or no real coverage. The report passes "
    "only if it says clearly that little or no coverage was found and does not invent stories. If it does "
    "that, count on_topic, in_window and covers_main_stories as true."
)


def judge(judge_llm, case: dict, sources: list, report: str) -> dict:
    listed = "\n\n".join(
        f"[{i + 1}] {s['title']}\nURL: {s['url']}\nPublished: {s['published'] or 'unknown'}\n{s['snippet']}"
        for i, s in enumerate(sources[:25])
    ) or "(no sources were found)"
    system = JUDGE_SYSTEM + (NO_COVERAGE_NOTE if case["kind"] == "no_coverage" else "")
    user = (f"Topic: {case['topic']}\nDate window: {case['start']} to {case['end']}\n\n"
            f"SOURCES:\n{listed}\n\nREPORT:\n{report}")
    verdict = with_schema(judge_llm, Verdict).invoke([("system", system), ("user", user)])
    return verdict.model_dump()


# --- Measuring a run ------------------------------------------------------------

def parse_sources(search_results: str) -> list:
    """Unique sources from collect_node's output (search_news and read_article blocks).

    Each source keeps the same text the summary step saw (a search snippet, or the full
    article text when the agent read it), so the judge grades against identical material.
    """
    sources = {}
    for block in search_results.split("\n---\n"):
        fields = {}
        for line in block.splitlines():
            for key in ("Source", "Published", "Title"):
                if line.startswith(key + ": ") and key not in fields:
                    fields[key] = line[len(key) + 2:].strip()
        url = fields.get("Source")
        if not url:
            continue
        text = block.split("Content: ", 1)[1].strip() if "Content: " in block else ""
        published = fields.get("Published", "")
        source = sources.setdefault(url, {"url": url, "title": url, "published": "", "snippet": ""})
        if fields.get("Title"):
            source["title"] = fields["Title"]
        if published and published != "unknown":
            source["published"] = published
        if len(text) > len(source["snippet"]):  # keep the fuller text (article over snippet)
            source["snippet"] = text[:agent.ARTICLE_CHAR_LIMIT]
    return list(sources.values())


def published_day(text: str):
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def run_case(graph, case: dict) -> dict:
    """Run the agent once and record what it did."""
    initial = {"topic": case["topic"], "time_period": f"{case['start']} to {case['end']}", "loop_count": 0}
    stats = {"searches": 0, "reads": 0, "agent_turns": 0, "drafts": 0, "critique_failures": 0}
    summary, search_results, critique_passed = "", "", None
    started = time.time()
    for chunk in graph.stream(initial, {"recursion_limit": agent.RECURSION_LIMIT}, stream_mode="updates"):
        for node, update in chunk.items():
            if not update:
                continue
            if node == "research_agent":
                stats["agent_turns"] += 1
                for call in update["messages"][-1].tool_calls:
                    stats["searches" if call["name"] == "search_news" else "reads"] += 1
            elif node == "create":
                stats["drafts"] += 1
                summary = update.get("summary", summary)
            elif node == "collect":
                search_results = update.get("search_results", search_results)
            elif node == "search_critique":
                critique_passed = update.get("critique_passed")
                if not update.get("critique_passed"):
                    stats["critique_failures"] += 1
    stats["seconds"] = round(time.time() - started, 1)
    return {"summary": summary, "search_results": search_results, "critique_passed": critique_passed, **stats}


def measure(case: dict, run: dict) -> dict:
    sources = parse_sources(run["search_results"])
    start, end = date.fromisoformat(case["start"]), date.fromisoformat(case["end"])
    dated = [d for d in (published_day(s["published"]) for s in sources) if d]
    in_window = [d for d in dated if start <= d <= end]
    return {
        "sources": sources,
        "n_sources": len(sources),
        "n_dated": len(dated),
        "pct_sources_in_window": round(100 * len(in_window) / len(dated)) if dated else None,
        "n_urls_cited": sum(1 for s in sources if s["url"] in run["summary"]),
    }


# --- Main -----------------------------------------------------------------------

def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["case_id"], row["mode"])] = row
    return done


def is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return "429" in text or "rate limit" in text or "rate-limit" in text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", help="Comma-separated case ids (default: all)")
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--out", help="Results folder (default: eval/results/<today>)")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE))
    args = parser.parse_args()

    test_set = json.loads((ROOT / "eval" / "test_set.json").read_text(encoding="utf-8"))
    cases = test_set["cases"]
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    modes = args.modes.split(",")
    out = Path(args.out) if args.out else ROOT / "eval" / "results" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    runs_file = out / "runs.jsonl"
    done = load_done(runs_file)

    graphs = {"with_check": agent.build_agent_graph(interactive=False, critique=True),
              "without_check": agent.build_agent_graph(interactive=False, critique=False)}
    judge_llm = get_llm(temperature=0, model=args.judge_model)
    agent_model = getattr(agent.llm, "model_name", "")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    # Alternate modes within each case, so changes in the news index over the run hit both equally.
    for case in cases:
        for mode in modes:
            if (case["id"], mode) in done:
                continue
            print(f"[{datetime.now():%H:%M:%S}] {case['id']} / {mode} ...", flush=True)
            row = {"case_id": case["id"], "mode": mode, "topic": case["topic"], "kind": case["kind"],
                   "start": case["start"], "end": case["end"], "agent_model": agent_model,
                   "judge_model": args.judge_model, "commit": commit,
                   "ran_at": datetime.now().isoformat(timespec="seconds")}
            for attempt in range(3):
                try:
                    run = run_case(graphs[mode], case)
                    row.update({k: v for k, v in run.items() if k != "search_results"})
                    row.update(measure(case, run))
                    row["error"] = None
                    break
                except Exception as e:  # keep going: a failed run is a result too
                    row["error"] = f"{type(e).__name__}: {e}"[:500]
                    if is_rate_limit(e) and attempt < 2:
                        print("    rate limited; waiting 60s", flush=True)
                        time.sleep(60)
                        continue
                    break

            judge_row(judge_llm, case, row, attempts=1)
            with runs_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"    {status_of(row)}  sources={row.get('n_sources')} drafts={row.get('drafts')} "
                  f"searches={row.get('searches')} {row.get('seconds')}s", flush=True)

    # Second pass: grade any report the judge couldn't reach the first time (free models get
    # rate-limited), with longer waits. Rewrites runs.jsonl with the new verdicts.
    rows = load_done(runs_file)
    by_id = {c["id"]: c for c in test_set["cases"]}
    missing = [r for r in rows.values() if not r.get("error") and r.get("summary") and not r.get("verdict")]
    for row in missing:
        print(f"[{datetime.now():%H:%M:%S}] judging {row['case_id']} / {row['mode']} ...", flush=True)
        row["judge_model"] = args.judge_model
        judge_row(judge_llm, by_id[row["case_id"]], row, attempts=5)
        print(f"    {status_of(row)}", flush=True)
    if missing:
        runs_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()),
                             encoding="utf-8")

    write_summary(out, test_set, load_done(runs_file))
    print(f"\nSummary: {out / 'summary.md'}")


def judge_row(judge_llm, case: dict, row: dict, attempts: int) -> None:
    """Add the judge's verdict and pass/fail to a run, retrying through rate limits."""
    if row.get("error") or not row.get("summary"):
        row["verdict"], row["passed"] = None, False
        return
    for attempt in range(attempts):
        try:
            row["verdict"] = judge(judge_llm, case, row["sources"], row["summary"])
            row["judge_error"] = None
            break
        except Exception as e:
            row["verdict"], row["judge_error"] = None, f"{type(e).__name__}: {e}"[:300]
            if attempt < attempts - 1:
                time.sleep(60 if is_rate_limit(e) else 5)
    verdict = row.get("verdict")
    row["passed"] = bool(verdict) and all(verdict.get(c) for c in CHECKS)


def status_of(row: dict) -> str:
    if row.get("error"):
        return "ERROR"
    if not row.get("verdict"):
        return "not judged yet"
    return "PASS" if row["passed"] else "fail: " + ", ".join(c for c in CHECKS if not row["verdict"].get(c))


def write_summary(out: Path, test_set: dict, rows: dict) -> None:
    def avg(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 1) if values else "–"

    lines = ["# Evaluation results", ""]
    any_row = next(iter(rows.values()), {})
    lines += [f"- Agent model: `{any_row.get('agent_model', '?')}`",
              f"- Judge model: `{any_row.get('judge_model', '?')}`",
              f"- Code: commit `{any_row.get('commit', '?')}`",
              f"- Cases: {len(test_set['cases'])}, one run per case per mode", ""]
    lines += ["| Mode | Pass rate | Errors | Avg sources | Avg % sources in window | Avg URLs cited | Avg drafts | Avg searches | Avg time (s) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mode in MODES:
        mode_rows = [r for (cid, m), r in rows.items() if m == mode]
        if not mode_rows:
            continue
        judged = [r for r in mode_rows if r.get("verdict") or r.get("error")]
        passed = sum(r["passed"] for r in judged)
        errors = sum(1 for r in mode_rows if r.get("error"))
        lines.append(
            f"| {mode} | {passed}/{len(judged)} ({round(100 * passed / len(judged)) if judged else 0}%) | {errors} | "
            f"{avg(r.get('n_sources') for r in mode_rows)} | {avg(r.get('pct_sources_in_window') for r in mode_rows)} | "
            f"{avg(r.get('n_urls_cited') for r in mode_rows)} | {avg(r.get('drafts') for r in mode_rows)} | "
            f"{avg(r.get('searches') for r in mode_rows)} | {avg(r.get('seconds') for r in mode_rows)} |")
    lines += ["", "## Per case", "",
              "| Case | Kind | with_check | without_check |", "| --- | --- | --- | --- |"]

    def cell(row):
        if not row:
            return "not run"
        if row.get("error"):
            return "ERROR"
        if not row.get("verdict"):
            return "not judged"
        failed = [c for c in CHECKS if not row["verdict"].get(c)]
        return "pass" if not failed else "fail: " + ", ".join(failed)

    for case in test_set["cases"]:
        lines.append(f"| {case['id']} | {case['kind']} | {cell(rows.get((case['id'], 'with_check')))} | "
                     f"{cell(rows.get((case['id'], 'without_check')))} |")
    lines += ["", "Pass = the run finished and the judge said yes to all five checks: "
              + ", ".join(CHECKS) + ". Full reports, sources and the judge's reasons are in `runs.jsonl`."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
