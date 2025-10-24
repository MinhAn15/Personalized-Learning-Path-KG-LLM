"""Simple in-memory session store used for demo sessions.

This is intentionally minimal: a dict keyed by UUID with simple session objects.
It provides create/get helpers and is safe for the dev/demo use-case.
"""
from typing import Dict, Any, Optional
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
