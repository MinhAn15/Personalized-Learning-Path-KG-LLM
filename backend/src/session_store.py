"""Simple in-memory session store used for demo sessions.

This is intentionally minimal: a dict keyed by UUID with simple session objects.
It provides create/get helpers and is safe for the dev/demo use-case.
"""
from typing import Any, Dict, Optional
from datetime import datetime
import uuid

_SESSIONS: Dict[str, Dict[str, Any]] = {}


def create_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + 'Z'
    session = {
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "payload": payload,
        "path": payload.get("path", []),
        "status": "created",
        "chat_history": [],
    }
    _SESSIONS[session_id] = session
    return session


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return _SESSIONS.get(session_id)


def update_session_path(session_id: str, path: Any) -> Optional[Dict[str, Any]]:
    s = _SESSIONS.get(session_id)
    if not s:
        return None
    s["path"] = path
    s["updated_at"] = datetime.utcnow().isoformat() + 'Z'
    s["status"] = "ready"
    return s


def append_chat_message(session_id: str, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    session = _SESSIONS.get(session_id)
    if not session:
        return None
    entry = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "timestamp": datetime.utcnow().isoformat() + 'Z',
    }
    history = session.setdefault("chat_history", [])
    history.append(entry)
    session["updated_at"] = entry["timestamp"]
    return entry


def get_chat_history(session_id: str, limit: Optional[int] = None) -> Optional[list]:
    session = _SESSIONS.get(session_id)
    if not session:
        return None
    history = session.get("chat_history", [])
    if limit is not None and limit > 0:
        return history[-limit:]
    return history


def update_session_metadata(session_id: str, updates: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not updates:
        return _SESSIONS.get(session_id)
    session = _SESSIONS.get(session_id)
    if not session:
        return None
    payload = session.setdefault("payload", {})
    payload.update(updates)
    session["updated_at"] = datetime.utcnow().isoformat() + 'Z'
    return session
