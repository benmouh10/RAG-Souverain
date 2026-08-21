"""Streamlit entry point for RAG Souverain.

Role-based navigation and page routing:
- Admin: Chat RAG, Files Indexation, Evaluation & Audit, Performance, System Parameters, User Management
- User:  Chat RAG, Indexed Files
"""

from __future__ import annotations

import logging

import streamlit as st

from app.app_ui import get_chroma_count, init_rag_engine
from app.css import hide_sidebar_style, inject_styles, sidebar_logo_html
from app.session import (
    allowed_pages,
    get_role,
    init_session_state,
    is_admin,
    set_current_page,
    sign_out,
)

logger = logging.getLogger("rag_ui")


def render_hero_header() -> None:
    st.markdown("<div class='hero-panel'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-badges'>"
        "<span class='status-badge'>Local-first</span>"
        "<span class='status-badge'>Private RAG</span>"
        "<span class='status-badge'>Fast results</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.title("RAG Souverain")
    st.caption("Local interface for document querying, evaluation, and knowledge base enrichment")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            "<div class='engine-status-card'><div class='label'>Engine status</div>"
            "<span class='status-value-green'>Ready</span></div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            "<div class='engine-status-card'><div class='label'>Vector store</div>"
            "<span class='status-value-green'>ChromaDB</span></div>",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            "<div class='engine-status-card'><div class='label'>Execution mode</div>"
            "<span class='status-value-green'>Local</span></div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar() -> None:
    """Sidebar with logo, profile pill, system health box, conversations switcher and actions."""
    st.sidebar.markdown(sidebar_logo_html(), unsafe_allow_html=True)
    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

    # User profile & role pill
    username = st.session_state.get("username", "unknown")
    role = get_role() or "user"
    role_class = "role-pill-admin" if role == "admin" else "role-pill-user"
    st.sidebar.markdown(
        f"<div class='sidebar-card' style='text-align:center'>"
        f"<span class='profile-pill'>👤 {username}</span> "
        f"<span class='role-pill {role_class}'>{role}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Conversations Section
    user_id = st.session_state.get("user_id")
    if user_id:
        from app.app_ui import _load_database

        Database = _load_database()
        db = Database()

        convs = db.list_conversations(user_id)
        if not convs:
            new_id = db.create_conversation(user_id, "New conversation")
            st.session_state["active_conversation_id"] = new_id
            convs = db.list_conversations(user_id)

        active_id = st.session_state.get("active_conversation_id")
        if not active_id or not any(c["conv_id"] == active_id for c in convs):
            active_id = convs[0]["conv_id"]
            st.session_state["active_conversation_id"] = active_id

        # 1. New Chat Button (Top)
        if st.sidebar.button("➕  New Chat", use_container_width=True, key="new_chat_top_btn"):
            new_id = db.create_conversation(user_id, "New conversation")
            st.session_state["active_conversation_id"] = new_id
            st.session_state["messages"] = []
            st.session_state["chat_history"] = []
            st.session_state.pop("history_loaded_conv_id", None)
            st.rerun()

        # 2. Options Expander (Right below New Chat)
        with st.sidebar.expander("⚙️ Options", expanded=False):
            current_conv = next((c for c in convs if c["conv_id"] == active_id), None)
            current_title = current_conv["title"] if current_conv else "New conversation"
            new_title = st.text_input("Rename Thread", value=current_title, key="rename_conv_title")
            if st.button("Save", use_container_width=True, key="save_conv_title_btn"):
                if new_title.strip():
                    db.rename_conversation(active_id, new_title.strip())
                    st.rerun()
            if len(convs) > 1 and st.button("🗑️ Delete Thread", use_container_width=True, key="del_conv_btn"):
                db.delete_conversation(active_id)
                st.session_state["active_conversation_id"] = None
                st.session_state["messages"] = []
                st.session_state["chat_history"] = []
                st.session_state.pop("history_loaded_conv_id", None)
                st.rerun()

        # 3. Recent Conversations Expander (Right below Options, collapsible)
        with st.sidebar.expander("💬 Recent Conversations", expanded=False):
            for c in convs:
                cid = c["conv_id"]
                is_active = (cid == active_id)
                title = c["title"] or "New conversation"
                display_title = f"{'▶ ' if is_active else ''}{title}"

                css_class = "conv-item-active" if is_active else "conv-item-btn"
                st.markdown(f"<div class='{css_class}'>", unsafe_allow_html=True)
                if st.button(display_title, key=f"conv_select_{cid}", use_container_width=True):
                    if cid != active_id:
                        st.session_state["active_conversation_id"] = cid
                        st.session_state["messages"] = []
                        st.session_state["chat_history"] = []
                        st.session_state.pop("history_loaded_conv_id", None)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

    # System health box
    try:
        rag = init_rag_engine()
        ollama_ok = False
        if getattr(rag, "llm", None) is not None:
            ollama_ok = rag.llm.health_check()
        else:
            ollama_ok = bool(rag.ollama_available)
        chroma_count = get_chroma_count(rag)
    except Exception:
        ollama_ok = False
        chroma_count = 0

    ollama_class = "status-badge status-green" if ollama_ok else "status-badge status-amber"
    chroma_class = "status-badge status-green" if chroma_count > 0 else "status-badge status-amber"
    ollama_label = "Online" if ollama_ok else "Degraded"
    st.sidebar.markdown("<div class='sidebar-section-title'>System Health</div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        f"<div class='sidebar-card sidebar-health'>"
        f"<div class='row'><span>Ollama</span><span class='{ollama_class}'>{ollama_label}</span></div>"
        f"<div class='row'><span>ChromaDB</span><span class='{chroma_class}'>Indexed</span></div>"
        f"<div class='row'><span>Indexed chunks</span><span class='chunk-count'>{chroma_count}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not ollama_ok:
        st.sidebar.warning("Ollama is not accessible. Start it with `ollama serve`.")

    st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

    # Actions
    st.sidebar.markdown(
        "<div class='sidebar-section-title'>Actions</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div class='sidebar-actions'><div class='reset-chat-btn'>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("↺ Reset Current Chat", use_container_width=True):
        reset_chat()
    st.sidebar.markdown("</div><div class='signout-btn'>", unsafe_allow_html=True)
    if st.sidebar.button("Sign Out", use_container_width=True):
        sign_out()
    st.sidebar.markdown("</div></div>", unsafe_allow_html=True)


def render_top_nav() -> str:
    """Horizontal pill/tab navigation in the main content area."""
    pages = allowed_pages()
    current = st.session_state.get("current_page", "Chat RAG")
    if current not in pages:
        current = pages[0]
    st.markdown("<div class='top-nav-bar'>", unsafe_allow_html=True)
    selected = st.segmented_control(
        "Page navigation",
        options=pages,
        default=current,
        label_visibility="collapsed",
        key="top_page_nav",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    selected = selected if selected in pages else current
    set_current_page(selected)
    return selected


def reset_chat() -> None:
    from app.app_ui import _load_database

    st.session_state["messages"] = []
    st.session_state["chat_history"] = []
    st.session_state.pop("history_loaded_conv_id", None)
    st.session_state.pop("cleanup_done", None)
    try:
        Database = _load_database()
        db = Database()
        active_conv_id = st.session_state.get("active_conversation_id")
        if active_conv_id:
            db.clear_chat_history(st.session_state.get("user_id"), conversation_id=active_conv_id)
    except Exception:
        logger.exception("Failed to clear chat history from DB")
    st.rerun()


def render_floating_scroll_buttons() -> None:
    st.markdown(
        '<a id="scroll-bottom-btn" href="#bottom-of-chat">↓</a>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<a id="scroll-top-btn" href="#top-of-page">↑</a>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Point d'entrée de l'interface Streamlit.

    Ordonnance le rendu : styles globaux, session, porte de connexion RBAC,
    navigation en bandeau et délégation vers la page sélectionnée.
    """
    st.set_page_config(page_title="RAG Souverain — UI", layout="wide")
    st.markdown('<a id="top-of-page"></a>', unsafe_allow_html=True)
    inject_styles()
    init_session_state()

    # Login gate — sidebar hidden on the auth view
    if not st.session_state.get("logged_in"):
        from app.pages.auth import render_login_page

        st.markdown(hide_sidebar_style(), unsafe_allow_html=True)
        render_login_page()
        return

    from app.app_ui import init_user_session

    user_id = st.session_state.get("user_id")
    init_user_session(user_id)

    active_conv_id = st.session_state.get("active_conversation_id")
    if user_id and active_conv_id:
        if st.session_state.get("history_loaded_conv_id") != active_conv_id:
            from app.app_ui import _load_database

            Database = _load_database()
            db = Database()
            saved = db.get_chat_history(user_id, conversation_id=active_conv_id, limit=200)
            st.session_state["messages"] = saved or []
            st.session_state["chat_history"] = list(st.session_state["messages"])
            st.session_state["history_loaded_conv_id"] = active_conv_id

    render_top_nav()
    render_hero_header()

    render_sidebar()

    # Floating scroll buttons only on the Chat page
    current = st.session_state.get("current_page", "Chat RAG")
    selected_page = current

    if selected_page == "Chat RAG":
        render_floating_scroll_buttons()

    if selected_page == "Chat RAG":
        from app.pages.chat import render_chat_page

        render_chat_page()
    elif selected_page == "Indexed Files" and not is_admin():
        from app.pages.indexed_files import render_indexed_files_page

        render_indexed_files_page()
    elif selected_page == "Files Indexation" and is_admin():
        from app.pages.file_indexation import render_file_indexation_page

        render_file_indexation_page()
    elif selected_page == "Evaluation & Audit" and is_admin():
        from app.pages.evaluation import render_evaluation_page

        render_evaluation_page()
    elif selected_page == "Performance" and is_admin():
        from app.pages.performance import render_performance_page

        render_performance_page()
    elif selected_page == "System Parameters" and is_admin():
        from app.pages.system_parameters import render_system_parameters_page

        render_system_parameters_page()
    elif selected_page == "User Management" and is_admin():
        from app.pages.users import render_users_page

        render_users_page()
    else:
        from app.pages.chat import render_chat_page

        render_chat_page()


if __name__ == "__main__":
    main()
