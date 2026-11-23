"""Saved topics and chat history for the GUI, kept as JSON files on disk.

Everything lives in data/ next to this file (gitignored). Set NEWS_AGENT_DATA_DIR
to keep it somewhere else.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

DATA_DIR = Path(os.getenv("NEWS_AGENT_DATA_DIR") or Path(__file__).parent / "data")
TOPICS_FILE = DATA_DIR / "topics.json"
CHATS_DIR = DATA_DIR / "chats"


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # write-then-rename, so a crash never leaves half a file


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def load_topics() -> list:
    return _read_json(TOPICS_FILE, [])


def save_topics(topics: list) -> None:
    _write_json(TOPICS_FILE, topics)


def load_chat(chat_id: str):
    return _read_json(CHATS_DIR / f"{chat_id}.json", None)


def save_chat(chat: dict) -> None:
    _write_json(CHATS_DIR / f"{chat['id']}.json", chat)


def list_chats() -> list:
    """Every saved chat, newest first."""
    chats = [_read_json(p, None) for p in CHATS_DIR.glob("*.json")]
    chats = [c for c in chats if c]
    return sorted(chats, key=lambda c: c.get("created_at", ""), reverse=True)
