"""Streamlit GUI for the News Research Agent.

Run with:  streamlit run app.py

Saved topics on the left; press Run and the agent's work streams in live, step by
step, next to a diagram of the graph that highlights where it is. When the report
is ready, the PR specialist can keep asking follow-up questions in a chat.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict

st.set_page_config(page_title="News Research Agent", page_icon="📰", layout="wide")

# Keep the column holding the diagram in view while the list of steps scrolls.
st.markdown(
    """<style>
    div[data-testid="stColumn"]:has(div.agent-sticky) {
        position: sticky; top: 3.5rem; align-self: flex-start;
    }
    </style>""",
    unsafe_allow_html=True,
)

try:
    import agent
    from evaluator import evaluate_topic
except Exception as e:  # e.g. missing OPENROUTER_API_KEY
    st.error(f"The agent could not start: {e}")
    st.stop()

import storage

PERIODS = ["Today", "Last week", "Last month", "Custom"]

# --- Graph diagrams ------------------------------------------------------------

RESEARCH_NODES = [
    ("research_brief", "Brief"),
    ("research_agent", "Agent\\ndecides next step"),
    ("tools", "Tools\\nsearch_news\\nread_article"),
    ("collect", "Collect sources"),
    ("create", "Summarize"),
    ("search_critique", "Editor check"),
    ("done", "Report"),
]
RESEARCH_EDGES = [
    ("research_brief", "research_agent", ""),
    ("research_agent", "tools", "calls a tool"),
    ("tools", "research_agent", "results"),
    ("research_agent", "collect", "has enough"),
    ("collect", "create", ""),
    ("create", "search_critique", ""),
    ("search_critique", "done", "passes"),
    ("search_critique", "research_brief", "needs work"),
]


def graph_dot(active: str | None = None, visited=()) -> str:
    """The research graph as Graphviz DOT. Yellow = step that just ran, green = already ran."""
    lines = [
        "digraph {",
        'bgcolor="transparent"; rankdir=TB; nodesep=0.3; ranksep=0.35;',
        'node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=14 margin="0.12,0.06" '
        'color="#868e96" fillcolor="#f1f3f5" fontcolor="#212529"];',
        'edge [color="#868e96" fontcolor="#868e96" fontname="Helvetica" fontsize=12];',
    ]
    for key, label in RESEARCH_NODES:
        attrs = f'label="{label}"'
        if key == active:
            attrs += ' fillcolor="#ffd43b" color="#e67700" penwidth=2'
        elif key in visited:
            attrs += ' fillcolor="#d3f9d8"'
        lines.append(f'"{key}" [{attrs}];')
    for a, b, label in RESEARCH_EDGES:
        lines.append(f'"{a}" -> "{b}"' + (f' [label=" {label}"]' if label else "") + ";")
    lines.append("}")
    return "\n".join(lines)


# --- Turning graph updates into readable steps ----------------------------------

ICONS = {
    "brief": "🧭", "retry": "🔁", "call_search": "🔎", "call_read": "📄", "result": "📥",
    "miss": "∅", "note": "💭", "stop": "✅", "limit": "⛔", "collect": "📦", "draft": "✍️",
    "critique_pass": "🟢", "critique_fail": "🟠", "critique_final": "⚠️", "error": "❌",
}

# One plain-English line per kind of step, for walking someone through a run.
EXPLAIN = {
    "brief": "Code hands the agent its task. From here, the model chooses every research step.",
    "retry": "The editor check failed, so its feedback goes back to the agent. The agent decides what extra research would fix it.",
    "call_search": "The model chose this query and date range itself. Nothing in the code fixed them.",
    "call_read": "The model judged a snippet too thin and chose to read the full article.",
    "result": "The tool's output goes back to the model, which reads it before choosing its next step.",
    "miss": "The tool came back empty. The model sees that and can try a different search.",
    "note": "Text the model wrote alongside its decision.",
    "stop": "The model judged it had enough and stopped calling tools.",
    "limit": f"The agent used all {agent.MAX_TOOL_ROUNDS} tool turns allowed per round, so the code made it stop.",
    "collect": "Plain code, not the model, gathers every tool result for the summary.",
    "draft": "A separate LLM call writes the executive summary from the collected sources.",
    "critique_pass": "A separate LLM call grades the draft for completeness, neutrality and relevance. It passed.",
    "critique_fail": "A separate LLM call grades the draft. It failed, so the agent gets another round.",
    "critique_final": "Still failing after 2 retries, so the run stops and keeps the latest draft. The report is flagged.",
    "error": "The run stopped. The message above says why.",
}


def text_of(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict))


def parse_search_results(text: str) -> list:
    """Split a search_news result back into {title, url, published} items."""
    items = []
    for block in text.split("\n---\n"):
        fields = {}
        for line in block.splitlines():
            for key in ("Source", "Published", "Title"):
                if line.startswith(key + ": "):
                    fields[key] = line[len(key) + 2:].strip()
        if fields.get("Source"):
            title = (fields.get("Title") or fields["Source"]).replace("[", "(").replace("]", ")")
            items.append({"url": fields["Source"], "title": title, "published": fields.get("Published", "")})
    return items


def events_from_update(node: str, update: dict, followup: bool = False) -> list:
    """Turn one LangGraph node update into the steps shown on screen."""
    events = []
    if node == "research_brief":
        messages = update.get("messages", [])
        if len(messages) == 1:  # a retry adds only the editor's feedback
            events.append({"node": node, "kind": "retry", "title": "Editor feedback sent back to the agent",
                           "body": text_of(messages[0])})
        else:
            events.append({"node": node, "kind": "brief", "title": "Task handed to the agent",
                           "body": text_of(messages[-1]).replace("\n", "  \n")})

    elif node == "research_agent":
        message = update["messages"][-1]
        text = text_of(message).strip()
        if message.tool_calls:
            if text:
                events.append({"node": node, "kind": "note", "title": "Agent's note", "body": text})
            for call in message.tool_calls:
                args = call.get("args", {})
                if call["name"] == "search_news":
                    events.append({"node": node, "kind": "call_search", "title": "Agent searches the news",
                                   "body": f"`{args.get('query', '')}` · {args.get('start_date', '?')} → {args.get('end_date', '?')}"})
                elif call["name"] == "read_article":
                    events.append({"node": node, "kind": "call_read", "title": "Agent reads a full article",
                                   "body": args.get("url", "")})
                else:
                    events.append({"node": node, "kind": "note", "title": f"Agent calls {call['name']}", "body": str(args)})
        else:
            if update.get("tool_rounds", 0) > agent.MAX_TOOL_ROUNDS:
                events.append({"node": node, "kind": "limit", "title": "Step limit reached"})
            if followup:
                events.append({"node": node, "kind": "stop", "title": "Agent has what it needs and answers"})
            else:
                events.append({"node": node, "kind": "stop", "title": "Agent decides it has enough",
                               "detail": text or None})

    elif node == "tools":
        for message in update.get("messages", []):
            content = text_of(message)
            if content.startswith(agent.TOOL_MISSES):
                events.append({"node": node, "kind": "miss", "title": content.splitlines()[0][:200]})
            elif message.name == "search_news":
                items = parse_search_results(content)
                events.append({"node": node, "kind": "result",
                               "title": f"{len(items)} article{'s' if len(items) != 1 else ''} found",
                               "items": items})
            else:
                url = content.splitlines()[0].replace("Source: ", "") if content else ""
                events.append({"node": node, "kind": "result",
                               "title": f"Article text fetched ({len(content):,} characters)", "body": url})

    elif node == "collect":
        results = update.get("search_results", "")
        count = 0 if results == "No results found." else results.count("Source: ")
        events.append({"node": node, "kind": "collect", "title": f"{count} sources collected for the summary",
                       "count": count})

    elif node == "create":
        events.append({"node": node, "kind": "draft", "title": "Draft summary written",
                       "detail": update.get("summary", "")})

    elif node == "search_critique":
        feedback = update.get("critique_feedback", "")
        if update.get("critique_passed"):
            events.append({"node": node, "kind": "critique_pass", "title": "Editor check passed"})
        elif update.get("is_satisfactory"):
            events.append({"node": node, "kind": "critique_final",
                           "title": "Editor check still failing after 2 retries; keeping the latest draft",
                           "body": feedback})
        else:
            events.append({"node": node, "kind": "critique_fail", "title": "Editor check failed", "body": feedback})
    return events


def render_event(event: dict, explain: bool) -> None:
    st.markdown(f"{ICONS.get(event['kind'], '•')} **{event['title']}**")
    if event.get("body"):
        st.markdown(event["body"])
    if event.get("items"):
        st.markdown("\n".join(
            f"- [{i['title']}]({i['url']})" + (f" · {i['published'][:16]}" if i.get("published") not in (None, "", "unknown") else "")
            for i in event["items"]
        ))
    if event.get("detail"):
        with st.container(height=160, border=True):  # scrollable; expanders can't nest here
            st.markdown(event["detail"])
    if explain and event["kind"] in EXPLAIN:
        st.caption(EXPLAIN[event["kind"]])


class LiveView:
    """Shows steps as they happen and keeps the diagram and counters in sync."""

    def __init__(self, status, graph_slot=None, metrics_slot=None, now_slot=None):
        # Write into a container inside the status box: leaving `with status:` would mark it complete.
        self.steps_box = status.container()
        self.graph_slot = graph_slot
        self.metrics_slot = metrics_slot
        self.now_slot = now_slot
        self.events = []
        self.visited = set()
        self.counts = {"Round": 1, "Searches": 0, "Articles read": 0, "Sources": 0}

    def add(self, event: dict) -> None:
        self.events.append(event)
        kind = event["kind"]
        if kind == "retry":
            self.counts["Round"] += 1
        elif kind == "call_search":
            self.counts["Searches"] += 1
        elif kind == "call_read":
            self.counts["Articles read"] += 1
        elif kind == "collect":
            self.counts["Sources"] = event.get("count", 0)
        with self.steps_box:
            render_event(event, st.session_state.get("explain", True))
        if self.now_slot is not None:
            # The newest step, kept next to the diagram so it's visible without scrolling.
            with self.now_slot.container(border=True):
                st.caption("Latest step")
                st.markdown(f"{ICONS.get(kind, '•')} **{event['title']}**")
                if st.session_state.get("explain", True) and kind in EXPLAIN:
                    st.caption(EXPLAIN[kind])
        self.refresh(event.get("node"))

    def refresh(self, active: str | None = None) -> None:
        if active:
            self.visited.add(active)
        if self.graph_slot is not None:
            self.graph_slot.graphviz_chart(graph_dot(active, self.visited), use_container_width=True)
        if self.metrics_slot is not None:
            show_metrics(self.metrics_slot, self.counts)


def show_metrics(slot, counts: dict) -> None:
    """Counters in a 2-by-2 grid, so the labels fit in a narrow column."""
    items = list(counts.items())
    with slot.container():
        for row in (items[:2], items[2:]):
            for col, (label, value) in zip(st.columns(2), row):
                col.metric(label, value)


# --- Topics and periods ---------------------------------------------------------

def resolve_period(topic: dict):
    """(start, end, label) for a saved topic. Relative periods are worked out on the day it runs."""
    today = date.today()
    period = topic["period"]
    if period == "Today":
        return today, today, "Today"
    if period == "Last week":
        return today - timedelta(days=7), today, "Last week"
    if period == "Last month":
        return today - timedelta(days=30), today, "Last month"
    start, end = date.fromisoformat(topic["start"]), date.fromisoformat(topic["end"])
    return start, end, "Custom"


def period_text(topic: dict) -> str:
    if topic["period"] == "Custom":
        return f"{topic['start']} → {topic['end']}"
    return topic["period"]


@st.cache_resource
def research_graph():
    return agent.build_agent_graph(interactive=False)


@st.cache_resource
def followup_graph():
    return agent.build_followup_graph()


# --- Running the agent ----------------------------------------------------------

def run_research(topic: dict) -> None:
    start, end, label = resolve_period(topic)
    time_period = f"{start} to {end}" if label == "Custom" else f"{label.lower()} ({start} to {end})"
    chat = {
        "id": storage.new_id(), "topic": topic["topic"], "period_label": label,
        "start": str(start), "end": str(end), "time_period": time_period,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": getattr(agent.llm, "model_name", ""), "followups": [],
    }

    st.header(topic["topic"])
    st.caption(f"{label} · {start} → {end} · model: {chat['model']}")
    left, right = st.columns([3, 2], gap="large")
    with right:
        st.markdown('<div class="agent-sticky"></div>', unsafe_allow_html=True)
        st.markdown("**Where the agent is**")
        now_slot = st.empty()
        graph_slot = st.empty()
        st.caption("Yellow: the step that just ran. Green: steps already run.")
        metrics_slot = st.empty()
    with left:
        status = st.status("The agent is researching…", expanded=True)

    view = LiveView(status, graph_slot, metrics_slot, now_slot)
    view.refresh()
    summary, search_results, passed, feedback = "", "", None, ""
    try:
        initial = {"topic": topic["topic"], "time_period": time_period, "loop_count": 0}
        for chunk in research_graph().stream(initial, {"recursion_limit": agent.RECURSION_LIMIT}, stream_mode="updates"):
            for node, update in chunk.items():
                if not update:
                    continue
                summary = update.get("summary", summary)
                search_results = update.get("search_results", search_results)
                if "critique_passed" in update:
                    passed, feedback = update["critique_passed"], update.get("critique_feedback", "")
                for event in events_from_update(node, update):
                    view.add(event)
        view.refresh("done")
        status.update(label="Research complete", state="complete")
        chat["status"] = "done"
    except Exception as e:
        view.add({"node": None, "kind": "error", "title": "The run stopped with an error", "body": str(e)})
        status.update(label="Research failed", state="error")
        chat["status"], chat["error"] = "error", str(e)

    chat.update(summary=summary, search_results=search_results, critique_passed=passed,
                critique_feedback=feedback, trace=view.events, stats=view.counts)
    storage.save_chat(chat)
    st.session_state.current_chat = chat["id"]
    st.session_state.pending_run = None
    st.rerun()


def run_followup(chat: dict, question: str) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    if chat.get("agent_messages"):
        history = messages_from_dict(chat["agent_messages"])
    else:
        history = [agent.followup_system_message(chat["topic"], chat["time_period"],
                                                 chat.get("summary", ""), chat.get("search_results", ""))]
    history.append(HumanMessage(content=question))

    with st.chat_message("assistant"):
        status = st.status("The agent is working on your question…", expanded=True)
        view = LiveView(status)
        new_messages, answer, error = [], "", None
        try:
            for chunk in followup_graph().stream({"messages": history, "tool_rounds": 0},
                                                 {"recursion_limit": agent.RECURSION_LIMIT}, stream_mode="updates"):
                for node, update in chunk.items():
                    if not update:
                        continue
                    new_messages.extend(update.get("messages", []))
                    for event in events_from_update(node, update, followup=True):
                        view.add(event)
            answers = [m for m in new_messages if isinstance(m, AIMessage) and not m.tool_calls]
            answer = text_of(answers[-1]) if answers else "(The agent returned no answer.)"
            status.update(label=f"Done in {len(view.events)} steps", state="complete", expanded=False)
        except Exception as e:
            error = str(e)
            view.add({"node": None, "kind": "error", "title": "The agent stopped with an error", "body": error})
            status.update(label="Failed", state="error")
        st.markdown(answer or f"Sorry, that failed: {error}")

    chat["followups"].append({"role": "user", "content": question})
    chat["followups"].append({"role": "assistant", "content": answer or f"Sorry, that failed: {error}",
                              "trace": view.events})
    if not error:
        chat["agent_messages"] = messages_to_dict(history + new_messages)
    storage.save_chat(chat)
    st.rerun()


# --- Pages ----------------------------------------------------------------------

def show_trace(events: list, title: str) -> None:
    with st.expander(title):
        for event in events:
            render_event(event, st.session_state.get("explain", True))


def show_chat(chat: dict) -> None:
    st.header(chat["topic"])
    st.caption(f"{chat['period_label']} · {chat['start']} → {chat['end']} · "
               f"run {chat['created_at'].replace('T', ' ')} · model: {chat.get('model') or 'unknown'}")

    if chat.get("stats"):
        show_metrics(st.empty(), chat["stats"])
    trace = chat.get("trace", [])
    with st.expander(f"How the agent produced this report ({len(trace)} steps)"):
        left, right = st.columns([3, 2], gap="large")
        with left:
            for event in trace:
                render_event(event, st.session_state.get("explain", True))
        with right:
            visited = {e["node"] for e in trace if e.get("node")}
            if chat.get("status") == "done":
                visited.add("done")
            st.graphviz_chart(graph_dot(None, visited), use_container_width=True)

    with st.chat_message("assistant"):
        if chat.get("status") == "error":
            st.error(f"The research run failed: {chat.get('error')}")
        else:
            if chat.get("critique_passed") is False:
                st.warning("The editor check still wasn't satisfied after 2 retries, so this is a "
                           f"best-effort draft. Its last feedback: {chat.get('critique_feedback') or 'none given'}")
            st.markdown(chat.get("summary") or "_No report was produced._")

    for message in chat.get("followups", []):
        with st.chat_message(message["role"]):
            if message.get("trace"):
                show_trace(message["trace"], f"How the agent answered ({len(message['trace'])} steps)")
            st.markdown(message["content"])

    if chat.get("status") == "done":
        question = st.chat_input("Ask the agent a follow-up question…")
        if question:
            run_followup(chat, question)


def show_welcome() -> None:
    st.title("News Research Agent")
    st.markdown(
        "Add a topic on the left, pick a time period, and press **Run**. "
        "You'll watch the agent work step by step, then you can chat with it about what it found."
    )
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            "**How it works**\n\n"
            "1. **Brief.** The code gives the agent the topic, the dates and today's date.\n"
            "2. **Agent ⇄ Tools.** The model decides each step: search the news "
            "(`search_news`, with a date range it picks), read a full article (`read_article`), or stop. "
            f"It gets up to {agent.MAX_TOOL_ROUNDS} tool turns per round.\n"
            "3. **Collect.** Code gathers everything the tools returned.\n"
            "4. **Summarize.** A separate LLM call writes the executive summary.\n"
            "5. **Editor check.** Another LLM call grades the summary. If it fails, the feedback "
            "goes back to the agent for another round (up to 2 retries).\n\n"
            "After the report, follow-up questions go to the same agent. It answers from the report "
            "and searches again only if it needs to."
        )
    with right:
        st.graphviz_chart(graph_dot(), use_container_width=True)


# --- Sidebar --------------------------------------------------------------------

def save_new_topic() -> None:
    text = st.session_state.get("new_topic", "").strip()
    period = st.session_state.get("new_period", "Last week")
    if not text:
        st.session_state.topic_msg = ("warning", "Type a topic first.")
        return
    topic = {"id": storage.new_id(), "topic": text, "period": period}
    if period == "Custom":
        picked = st.session_state.get("new_range") or ()
        if len(picked) != 2:
            st.session_state.topic_msg = ("warning", "Pick both a start and an end date.")
            return
        topic["start"], topic["end"] = str(picked[0]), str(picked[1])
    try:
        check = evaluate_topic(text)
    except Exception as e:
        st.session_state.topic_msg = ("error", f"Couldn't check the topic: {e}")
        return
    if not check.is_valid:
        st.session_state.topic_msg = ("warning", f"The topic checker rejected this: {check.feedback}")
        return
    corrected = (check.corrected_value or text).strip()
    topic["topic"] = corrected
    storage.save_topics(storage.load_topics() + [topic])
    note = f" The checker corrected it to “{corrected}”." if corrected != text else ""
    st.session_state.topic_msg = ("success", f"Saved.{note}")
    st.session_state.new_topic = ""


def delete_topic(topic_id: str) -> None:
    storage.save_topics([t for t in storage.load_topics() if t["id"] != topic_id])


def start_run(topic_id: str) -> None:
    st.session_state.pending_run = topic_id
    st.session_state.current_chat = None


def open_chat(chat_id: str) -> None:
    st.session_state.current_chat = chat_id
    st.session_state.pending_run = None


def sidebar() -> None:
    with st.sidebar:
        st.title("📰 News Research")
        st.toggle("Explain each step", value=True, key="explain",
                  help="Adds a plain-English line under each step, for walking someone through a run.")

        st.subheader("New topic")
        st.text_input("Topic", key="new_topic", placeholder="e.g. Acme Corp product recall")
        st.selectbox("Time period", PERIODS, index=1, key="new_period")
        if st.session_state.get("new_period") == "Custom":
            st.date_input("Dates", key="new_range", max_value=date.today(),
                          value=(date.today() - timedelta(days=7), date.today()))
        st.button("Save topic", on_click=save_new_topic, use_container_width=True)
        message = st.session_state.pop("topic_msg", None)
        if message:
            getattr(st, message[0])(message[1])

        st.subheader("Saved topics")
        topics = storage.load_topics()
        if not topics:
            st.caption("No topics yet.")
        for topic in topics:
            with st.container(border=True):
                st.markdown(f"**{topic['topic']}**  \n{period_text(topic)}")
                run_col, delete_col = st.columns(2)
                run_col.button("▶ Run", key=f"run_{topic['id']}", type="primary", use_container_width=True,
                               on_click=start_run, args=(topic["id"],))
                delete_col.button("Delete", key=f"del_{topic['id']}", use_container_width=True,
                                  on_click=delete_topic, args=(topic["id"],))

        st.subheader("Previous chats")
        chats = storage.list_chats()
        if not chats:
            st.caption("No chats yet.")
        for chat in chats:
            when = chat.get("created_at", "")[:16].replace("T", " ")
            current = chat["id"] == st.session_state.get("current_chat")
            st.button(f"{chat['topic']} · {when}", key=f"chat_{chat['id']}", use_container_width=True,
                      type="primary" if current else "secondary", on_click=open_chat, args=(chat["id"],))


# --- Main -----------------------------------------------------------------------

st.session_state.setdefault("current_chat", None)
st.session_state.setdefault("pending_run", None)
sidebar()

pending = st.session_state.pending_run
if pending:
    topic = next((t for t in storage.load_topics() if t["id"] == pending), None)
    if topic:
        run_research(topic)
    else:
        st.session_state.pending_run = None
        show_welcome()
elif st.session_state.current_chat and storage.load_chat(st.session_state.current_chat):
    show_chat(storage.load_chat(st.session_state.current_chat))
else:
    show_welcome()
