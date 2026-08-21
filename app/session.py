"""Session state management for the Streamlit UI.

Tracks authentication, user role, landing page, and navigation for
the role-based interface (RBAC).
"""

from __future__ import annotations

import secrets

import streamlit as st

# Page registry
ADMIN_PAGES = [
    "Chat RAG",
    "Files Indexation",
    "Evaluation & Audit",
    "Performance",
    "System Parameters",
    "User Management",
]
USER_PAGES = [
    "Chat RAG",
    "Indexed Files",
]


def init_session_state() -> None:
    """Seed the session state with the keys used across the app."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = None
    if "auth_token" not in st.session_state:
        st.session_state["auth_token"] = None
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Chat RAG"
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = list(st.session_state["messages"])
    elif not st.session_state["messages"] and st.session_state["chat_history"]:
        st.session_state["messages"] = list(st.session_state["chat_history"])
    if "eval_running" not in st.session_state:
        st.session_state["eval_running"] = False
    if "uploader_id" not in st.session_state:
        st.session_state["uploader_id"] = 0
    if "llm_temperature" not in st.session_state:
        st.session_state["llm_temperature"] = 0.1
    if "top_k" not in st.session_state:
        st.session_state["top_k"] = 4
    if "chunk_size" not in st.session_state:
        st.session_state["chunk_size"] = 512
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = "mistral"
    if "active_conversation_id" not in st.session_state:
        st.session_state["active_conversation_id"] = None


def sign_in(user_id: int, username: str, role: str) -> None:
    """Record a successful authentication in the session state."""
    st.session_state["logged_in"] = True
    st.session_state["user_id"] = user_id
    st.session_state["username"] = username
    st.session_state["user_role"] = role
    st.session_state["role"] = role
    st.session_state["auth_token"] = secrets.token_hex(16)
    st.session_state["current_page"] = "Chat RAG"
    st.session_state.pop("rag_engine", None)
    st.session_state.pop("history_loaded", None)
    st.session_state.pop("cleanup_done", None)


def sign_out() -> None:
    """Clear all session data and return to the login screen."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()
    st.rerun()


def get_role() -> str | None:
    return st.session_state.get("user_role") or st.session_state.get("role")


def is_admin() -> bool:
    return get_role() == "admin"


def require_admin() -> bool:
    """Garde serveur (M1) : vérifie le rôle administrateur côté session.

    Chaque page/action sensible (indexation, suppression, purge, paramètres,
    gestion des utilisateurs, évaluation) doit ré-appeler cette garde, en plus
    du filtrage de navigation, afin de ne jamais exposer une action
    administrative si la session n'est pas authentifiée comme admin.
    """
    if not is_admin():
        import streamlit as st

        st.error("Accès refusé — privilèges administrateur requis.")
        return False
    return True


def allowed_pages() -> list[str]:
    return ADMIN_PAGES if is_admin() else USER_PAGES


def set_current_page(page: str) -> None:
    if page in allowed_pages():
        st.session_state["current_page"] = page
