"""Briefleaf: the web app for the News Research Agent.

Run with:  streamlit run app.py

Topics screen: save the subjects you track and run one. While the agent works,
every step it takes streams in live. When the report is ready, keep asking
follow-up questions in the chat. Previous chats are listed in the sidebar.
"""
from __future__ import annotations

import hmac
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, messages_from_dict, messages_to_dict

APP_NAME = "Briefleaf"
ASSETS = Path(__file__).parent / "assets"
LOGO = str(ASSETS / "briefleaf-logo.svg")
MARK = str(ASSETS / "briefleaf-mark.svg")
MARK_PNG = str(ASSETS / "briefleaf-mark.png")

st.set_page_config(page_title=APP_NAME, page_icon=MARK_PNG, layout="wide")
st.logo(LOGO, icon_image=MARK, size="large")

# Colours and the theme live in .streamlit/config.toml; this is layout polish only.
st.markdown(
    """<style>
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2.5rem; max-width: 1100px; }
    /* Sidebar chat list: one line per chat, cut off with an ellipsis. */
    section[data-testid="stSidebar"] .stButton button { justify-content: flex-start; }
    section[data-testid="stSidebar"] .stButton button p {
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: left;
    }
    section[data-testid="stSidebar"] .stButton button div { min-width: 0; }
    .bl-eyebrow {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
        color: #5c7a65; margin: 1.2rem 0 0.4rem;
    }
    .bl-subtle { color: #5c7a65; }
    </style>""",
    unsafe_allow_html=True,
)


def setting(name: str) -> str:
    """A setting from the environment (.env locally) or from Secrets on Streamlit Community Cloud."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # no secrets file, which is normal when running locally
        return ""


def require_password() -> None:
    """Ask for APP_PASSWORD before showing anything. Skipped when no password is set.

    On a public deployment this stops strangers from spending your OpenRouter and Tavily quota.
    """
    expected = setting("APP_PASSWORD")
    if not expected or st.session_state.get("signed_in"):
        return
    _, middle, _ = st.columns([1, 1.2, 1])
    with middle:
        st.write("")
        st.image(LOGO, width=220)
        st.markdown("#### Sign in")
        with st.form("sign_in"):
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
        if submitted:
            if hmac.compare_digest(password.encode(), expected.encode()):
                st.session_state.signed_in = True
                st.rerun()
            st.error("That password isn't right.")
    st.stop()


require_password()

try:
    import agent
    from evaluator import evaluate_topic
except Exception as e:  # e.g. missing OPENROUTER_API_KEY
    st.error(f"The agent could not start: {e}")
    st.stop()

import storage

PERIODS = ["Today", "Last week", "Last month", "Custom"]
USER_AVATAR = ":material/person:"

# --- Turning graph updates into readable steps ----------------------------------

ICONS = {
    "brief": ":material/assignment:", "retry": ":material/replay:", "call_search": ":material/search:",
    "call_read": ":material/article:", "result": ":material/library_books:", "miss": ":material/search_off:",
    "note": ":material/chat_bubble:", "stop": ":material/task_alt:", "limit": ":material/block:",
    "collect": ":material/inventory_2:", "draft": ":material/edit_note:", "critique_pass": ":material/verified:",
    "critique_fail": ":material/rate_review:", "critique_final": ":material/warning:", "error": ":material/error:",
}
WARNING_KINDS = {"critique_fail", "critique_final", "limit", "retry"}


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
                    events.append({"node": node, "kind": "call_search", "title": "Searching the news",
                                   "body": f"`{args.get('query', '')}` · {args.get('start_date', '?')} → {args.get('end_date', '?')}"})
                elif call["name"] == "read_article":
                    events.append({"node": node, "kind": "call_read", "title": "Reading a full article",
                                   "body": args.get("url", "")})
                else:
                    events.append({"node": node, "kind": "note", "title": f"Calling {call['name']}", "body": str(args)})
        else:
            if update.get("tool_rounds", 0) > agent.MAX_TOOL_ROUNDS:
                events.append({"node": node, "kind": "limit", "title": "Step limit reached"})
            if followup:
                events.append({"node": node, "kind": "stop", "title": "Writing the answer"})
            else:
                events.append({"node": node, "kind": "stop", "title": "Research complete",
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


def render_event(event: dict) -> None:
    kind = event["kind"]
    color = "red" if kind == "error" else "orange" if kind in WARNING_KINDS else "green"
    st.markdown(f":{color}[{ICONS.get(kind, ':material/circle:')}] **{event['title']}**")
    if event.get("body"):
        st.markdown(event["body"])
    if event.get("items"):
        st.markdown("\n".join(
            f"- [{i['title']}]({i['url']})" + (f" · {i['published'][:10]}" if i.get("published") not in (None, "", "unknown") else "")
            for i in event["items"]
        ))
    if event.get("detail"):
        with st.container(height=160, border=True):  # scrollable; expanders can't nest here
            st.markdown(event["detail"])


class LiveView:
    """Writes steps into a status box as they happen and keeps running counts."""

    def __init__(self, status):
        # Write into a container inside the status box: leaving `with status:` would mark it complete.
        self.steps_box = status.container()
        self.events = []
        self.counts = {"rounds": 1, "searches": 0, "reads": 0, "sources": 0}

    def add(self, event: dict) -> None:
        self.events.append(event)
        kind = event["kind"]
        if kind == "retry":
            self.counts["rounds"] += 1
        elif kind == "call_search":
            self.counts["searches"] += 1
        elif kind == "call_read":
            self.counts["reads"] += 1
        elif kind == "collect":
            self.counts["sources"] = event.get("count", 0)
        with self.steps_box:
            render_event(event)


def stats_text(stats: dict) -> str:
    def n(count, word, plural=None):
        return f"{count} {word if count == 1 else (plural or word + 's')}"
    # Older chats saved these under display labels.
    rounds = stats.get("rounds", stats.get("Round", 1))
    searches = stats.get("searches", stats.get("Searches", 0))
    reads = stats.get("reads", stats.get("Articles read", 0))
    sources = stats.get("sources", stats.get("Sources", 0))
    return " · ".join([n(rounds, "round"), n(searches, "search", "searches"), n(reads, "article") + " read", n(sources, "source")])


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
    return date.fromisoformat(topic["start"]), date.fromisoformat(topic["end"]), "Custom"


def period_text(topic: dict) -> str:
    if topic["period"] == "Custom":
        return f"{topic['start']} → {topic['end']}"
    return topic["period"]


def short(text: str, limit: int = 30) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@st.cache_resource
def research_graph():
    return agent.build_agent_graph(interactive=False)


@st.cache_resource
def followup_graph():
    return agent.build_followup_graph()


# --- Navigation -----------------------------------------------------------------

def go_topics() -> None:
    st.session_state.view = "topics"
    st.session_state.current_chat = None
    st.session_state.pending_run = None


def start_run(topic_id: str) -> None:
    st.session_state.view = "run"
    st.session_state.pending_run = topic_id
    st.session_state.current_chat = None


def open_chat(chat_id: str) -> None:
    st.session_state.view = "chat"
    st.session_state.current_chat = chat_id
    st.session_state.pending_run = None


def delete_chat(chat_id: str) -> None:
    storage.delete_chat(chat_id)
    if st.session_state.get("current_chat") == chat_id:
        go_topics()


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
        st.session_state.topic_msg = ("warning", f"This topic needs to be more specific: {check.feedback}")
        return
    corrected = (check.corrected_value or text).strip()
    topic["topic"] = corrected
    storage.save_topics(storage.load_topics() + [topic])
    note = f" Corrected to “{corrected}”." if corrected != text else ""
    st.session_state.topic_msg = ("success", f"Topic saved.{note}")
    st.session_state.new_topic = ""


def delete_topic(topic_id: str) -> None:
    storage.save_topics([t for t in storage.load_topics() if t["id"] != topic_id])


# --- Running the agent ----------------------------------------------------------

def run_research(topic: dict) -> None:
    start, end, label = resolve_period(topic)
    time_period = f"{start} to {end}" if label == "Custom" else f"{label.lower()} ({start} to {end})"
    chat = {
        "id": storage.new_id(), "topic_id": topic["id"], "topic": topic["topic"], "period_label": label,
        "start": str(start), "end": str(end), "time_period": time_period,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": getattr(agent.llm, "model_name", ""), "followups": [],
    }

    st.title(topic["topic"])
    st.caption(f"{label} · {start} → {end}")
    status = st.status("Researching…", expanded=True)
    view = LiveView(status)
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
        status.update(label="Research complete", state="complete")
        chat["status"] = "done"
    except Exception as e:
        view.add({"node": None, "kind": "error", "title": "The run stopped with an error", "body": str(e)})
        status.update(label="Research failed", state="error")
        chat["status"], chat["error"] = "error", str(e)

    chat.update(summary=summary, search_results=search_results, critique_passed=passed,
                critique_feedback=feedback, trace=view.events, stats=view.counts)
    storage.save_chat(chat)
    open_chat(chat["id"])
    st.rerun()


def run_followup(chat: dict, question: str) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    if chat.get("agent_messages"):
        history = messages_from_dict(chat["agent_messages"])
    else:
        history = [agent.followup_system_message(chat["topic"], chat["time_period"],
                                                 chat.get("summary", ""), chat.get("search_results", ""))]
    history.append(HumanMessage(content=question))

    with st.chat_message("assistant", avatar=MARK_PNG):
        status = st.status("Working on it…", expanded=True)
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


# --- Screens --------------------------------------------------------------------

def show_topics() -> None:
    st.title("Topics")
    st.markdown('<p class="bl-subtle">Save the subjects you track. Run one to get a fresh coverage '
                "report, then ask the agent about it.</p>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("**Add a topic**")
        topic_col, period_col, button_col = st.columns([3, 1.4, 1], vertical_alignment="bottom")
        topic_col.text_input("Topic", key="new_topic", placeholder="e.g. Acme Corp product recall")
        period_col.selectbox("Time period", PERIODS, index=1, key="new_period")
        button_col.button("Save topic", type="primary", icon=":material/add:", width="stretch",
                          on_click=save_new_topic)
        if st.session_state.get("new_period") == "Custom":
            st.date_input("Dates", key="new_range", max_value=date.today(),
                          value=(date.today() - timedelta(days=7), date.today()))
        message = st.session_state.pop("topic_msg", None)
        if message:
            getattr(st, message[0])(message[1])

    st.markdown('<div class="bl-eyebrow">Saved topics</div>', unsafe_allow_html=True)
    topics = storage.load_topics()
    if not topics:
        with st.container(border=True):
            st.markdown(":green[:material/eco:] No topics yet. Add the first one above.")
        return

    chats = storage.list_chats()
    columns = st.columns(3)
    for i, topic in enumerate(topics):
        last = next((c for c in chats if c.get("topic_id") == topic["id"] or c.get("topic") == topic["topic"]), None)
        last_run = f"Last run {last['created_at'][:16].replace('T', ' ')}" if last else "Not run yet"
        with columns[i % 3]:
            with st.container(border=True):
                st.markdown(f"**{topic['topic']}**")
                st.caption(f":material/calendar_today: {period_text(topic)}  \n{last_run}")
                run_col, delete_col = st.columns([3, 1], vertical_alignment="center")
                run_col.button("Run", key=f"run_{topic['id']}", type="primary", icon=":material/play_arrow:",
                               width="stretch", on_click=start_run, args=(topic["id"],))
                delete_col.button("", key=f"del_{topic['id']}", icon=":material/delete:", type="tertiary",
                                  help="Delete topic", on_click=delete_topic, args=(topic["id"],))


def show_chat(chat: dict) -> None:
    st.title(chat["topic"])
    stats = f" · {stats_text(chat['stats'])}" if chat.get("stats") else ""
    st.caption(f"{chat['period_label']} · {chat['start']} → {chat['end']} · "
               f"{chat['created_at'][:16].replace('T', ' ')}{stats}")

    trace = chat.get("trace", [])
    with st.expander(f"How the agent produced this report ({len(trace)} steps)", icon=":material/account_tree:"):
        for event in trace:
            render_event(event)

    with st.chat_message("assistant", avatar=MARK_PNG):
        if chat.get("status") == "error":
            st.error(f"The research run failed: {chat.get('error')}")
        else:
            if chat.get("critique_passed") is False:
                st.warning("The editor check still wasn't satisfied after 2 retries, so this is a "
                           f"best-effort draft. Its last feedback: {chat.get('critique_feedback') or 'none given'}")
            st.markdown(chat.get("summary") or "_No report was produced._")

    for message in chat.get("followups", []):
        avatar = MARK_PNG if message["role"] == "assistant" else USER_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            if message.get("trace"):
                with st.expander(f"How the agent answered ({len(message['trace'])} steps)"):
                    for event in message["trace"]:
                        render_event(event)
            st.markdown(message["content"])

    if chat.get("status") == "done":
        question = st.chat_input("Ask a follow-up question…")
        if question:
            run_followup(chat, question)


def sidebar() -> None:
    with st.sidebar:
        st.button("Topics", icon=":material/topic:", width="stretch",
                  type="primary" if st.session_state.view == "topics" else "secondary", on_click=go_topics)

        st.markdown('<div class="bl-eyebrow">Previous chats</div>', unsafe_allow_html=True)
        chats = storage.list_chats()
        if not chats:
            st.caption("Your research runs will appear here.")
        for chat in chats:
            when = chat.get("created_at", "")[:16].replace("T", " ")
            current = chat["id"] == st.session_state.get("current_chat")
            title_col, delete_col = st.columns([5, 1], gap="small", vertical_alignment="center")
            title_col.button(short(chat["topic"]), key=f"chat_{chat['id']}", width="stretch",
                             type="primary" if current else "secondary", help=f"{chat['topic']} · {when}",
                             on_click=open_chat, args=(chat["id"],))
            delete_col.button("", key=f"delchat_{chat['id']}", icon=":material/delete:", type="tertiary",
                              help="Delete chat", on_click=delete_chat, args=(chat["id"],))


# --- Main -----------------------------------------------------------------------

st.session_state.setdefault("view", "topics")
st.session_state.setdefault("current_chat", None)
st.session_state.setdefault("pending_run", None)
sidebar()

view = st.session_state.view
if view == "run" and st.session_state.pending_run:
    topic = next((t for t in storage.load_topics() if t["id"] == st.session_state.pending_run), None)
    if topic:
        run_research(topic)
    else:
        go_topics()
        show_topics()
elif view == "chat" and st.session_state.current_chat and storage.load_chat(st.session_state.current_chat):
    show_chat(storage.load_chat(st.session_state.current_chat))
else:
    show_topics()
