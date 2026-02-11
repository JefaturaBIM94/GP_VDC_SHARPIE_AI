from __future__ import annotations

# Thin re-export layer so `backend.main` can import from `backend.session_store`
# while the implementation lives at repo root (`session_store.py`).

from session_store import (  # noqa: F401
    DATA_DIR,
    IMAGES_DIR,
    SESSIONS_DIR,
    SessionStore,
    SegmentResult,
    create_session,
    load_session,
    save_session,
    list_sessions,
)

