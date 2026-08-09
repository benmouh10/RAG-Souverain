"""Indexed Files page — read-only document catalog for normal users."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import streamlit as st

from app.app_ui import aggregate_sources, delete_file_from_vector_store, init_rag_engine
from app.session import is_admin

logger = logging.getLogger("rag_ui")


def _file_type_label(source_file: str) -> str:
    suffix = Path(source_file).suffix.lower()
    return suffix if suffix else "—"


def render_indexed_files_page() -> None:
    rag = init_rag_engine()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>Indexed Files</div>", unsafe_allow_html=True)
    st.caption("All documents available in the knowledge base, with their chunk counts.")
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        sources = aggregate_sources(rag.vector_store.list_sources())
    except Exception as e:
        st.error(f"Cannot read the vector store: {type(e).__name__}: {e}")
        return

    if not sources:
        st.info("No documents indexed yet. Check back once an administrator indexes files.")
        return

    total_chunks = sum(s["chunk_count"] for s in sources)
    total_files = len(sources)

    c1, c2 = st.columns(2)
    c1.metric("Indexed Files", total_files)
    c2.metric("Total Chunks", total_chunks)

    admin_view = is_admin()

    header_cols = st.columns([3, 1, 1, 1.2, 1.4] if admin_view else [3, 1, 1, 1.2])
    labels = ["File Name", "Type", "Chunks", "Status"]
    if admin_view:
        labels.append("Action")
    for col, label in zip(header_cols, labels):
        col.markdown(f"<div class='file-table-th'>{label}</div>", unsafe_allow_html=True)

    for i, source in enumerate(sources):
        name = source["source_file"]
        file_type = _file_type_label(name)
        chunk_count = int(source.get("chunk_count", 0))

        if admin_view:
            col_name, col_type, col_chunks, col_status, col_action = st.columns(
                [3, 1, 1, 1.2, 1.4]
            )
        else:
            col_name, col_type, col_chunks, col_status = st.columns([3, 1, 1, 1.2])
            col_action = None

        col_name.markdown(f"<strong>{html.escape(name)}</strong>", unsafe_allow_html=True)
        col_type.markdown(
            f"<span class='file-type'>{html.escape(file_type)}</span>",
            unsafe_allow_html=True,
        )
        col_chunks.markdown(
            f"<span class='chunk-count'>{chunk_count}</span>",
            unsafe_allow_html=True,
        )
        col_status.markdown(
            "<span class='status-badge status-green'>Indexed</span>",
            unsafe_allow_html=True,
        )

        if admin_view and col_action is not None and col_action.button(
            "🗑️ Delete File", key=f"del_idx_{i}", use_container_width=True
        ):
            try:
                deleted = delete_file_from_vector_store(name, rag=rag)
                st.success(f"{deleted} chunk(s) removed from '{name}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {type(e).__name__}: {e}")
                logger.exception("Delete error for %s", name)
