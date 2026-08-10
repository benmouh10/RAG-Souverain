"""Files Indexation page — admin-only upload & indexing dashboard."""

from __future__ import annotations

import html
import logging
from pathlib import Path

import streamlit as st

from app.app_ui import (
    IngestError,
    OllamaConnectionError,
    aggregate_sources,
    delete_file_from_vector_store,
    get_upload_widget_key,
    ingest_uploaded_file,
    init_rag_engine,
    reset_upload_widget_state,
)
from app.session import require_admin

logger = logging.getLogger("rag_ui")


def _file_type_label(source_file: str) -> str:
    suffix = Path(source_file).suffix.lower()
    return suffix if suffix else "—"


def render_file_indexation_page() -> None:
    if not require_admin():
        return

    rag = init_rag_engine()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Files Indexation")
    st.caption("Upload and index documents (PDF, TXT, MD, CSV) into the local vector store.")
    st.markdown("</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Files (PDF, TXT, MD, CSV)",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Select one or more documents to index into ChromaDB in a single batch",
        key=get_upload_widget_key(),
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected.")

    if st.button("Index file(s)", use_container_width=True):
        if not uploaded_files:
            st.warning("Please select at least one file first.")
        else:
            total_added = 0
            total_indexed = 0
            failures: list[tuple[str, str]] = []
            progress = st.progress(0.0, text="Ready to index...")

            for i, uploaded_file in enumerate(uploaded_files, start=1):
                progress.progress(
                    (i - 1) / len(uploaded_files),
                    text=f"Indexing ({i}/{len(uploaded_files)}): {uploaded_file.name}",
                )
                try:
                    added, file_name = ingest_uploaded_file(uploaded_file, rag=rag)
                    total_added += added
                    total_indexed += 1
                except IngestError as e:
                    failures.append((uploaded_file.name, str(e)))
                    logger.warning("Ingest error for %s: %s", uploaded_file.name, e)
                except OllamaConnectionError as e:
                    failures.append((uploaded_file.name, str(e)))
                except Exception as e:
                    failures.append((uploaded_file.name, type(e).__name__))
                    logger.exception("Unexpected ingest error for %s", uploaded_file.name)

            progress.progress(1.0, text="Batch complete.")
            progress.empty()

            if total_indexed:
                st.success(
                    f"{total_added} chunk(s) indexed from {total_indexed} file(s)."
                )
                st.session_state["upload_status"] = (
                    f"{total_added} chunk(s) indexed from {total_indexed} file(s)."
                )
            if failures:
                st.error("Some files could not be indexed:")
                for name, reason in failures:
                    st.markdown(f"- **{name}** — {reason}")
            reset_upload_widget_state()

    if st.checkbox("Show last indexing status", value=False):
        st.info(st.session_state.get("upload_status") or "No files indexed yet.")

    st.markdown("---")
    st.subheader("Indexed documents")
    st.caption("Files currently stored in the shared vector store. Use Delete to remove one file.")

    try:
        sources = aggregate_sources(rag.vector_store.list_sources())
    except Exception as e:
        st.error(f"Cannot read the vector store: {type(e).__name__}: {e}")
        sources = []

    if not sources:
        st.info("No documents indexed yet.")
    else:
        total_chunks = sum(s["chunk_count"] for s in sources)
        c1, c2 = st.columns(2)
        c1.metric("Indexed Files", len(sources))
        c2.metric("Total Chunks", total_chunks)

        header_cols = st.columns([3, 1, 1, 1.2, 1.4])
        for col, label in zip(header_cols, ["File Name", "Type", "Chunks", "Status", "Action"]):
            col.markdown(f"<div class='file-table-th'>{label}</div>", unsafe_allow_html=True)

        for i, source in enumerate(sources):
            name = source["source_file"]
            file_type = _file_type_label(name)
            chunk_count = int(source.get("chunk_count", 0))
            col_name, col_type, col_chunks, col_status, col_action = st.columns([3, 1, 1, 1.2, 1.4])
            col_name.markdown(f"<strong>{html.escape(name)}</strong>", unsafe_allow_html=True)
            col_type.markdown(f"<span class='file-type'>{html.escape(file_type)}</span>", unsafe_allow_html=True)
            col_chunks.markdown(f"<span class='chunk-count'>{chunk_count}</span>", unsafe_allow_html=True)
            col_status.markdown(
                "<span class='status-badge status-green'>Indexed</span>", unsafe_allow_html=True
            )
            if col_action.button("🗑️ Delete File", key=f"del_file_{i}", use_container_width=True):
                try:
                    deleted = delete_file_from_vector_store(name, rag=rag)
                    st.session_state["upload_status"] = (
                        f"{deleted} chunk(s) removed from '{name}'."
                    )
                    st.success(f"{deleted} chunk(s) removed from '{name}'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Delete failed: {type(e).__name__}: {e}")
                    logger.exception("Delete error for %s", name)

    st.markdown("---")
    st.subheader("Purge vector store")
    st.caption("Remove every chunk from the collection to start fresh.")
    if st.button("Purge vector store", use_container_width=True):
        try:
            purged_count = rag.vector_store.clear()
            st.session_state["upload_status"] = f"Vector store purged. {purged_count} element(s) remaining."
            st.success("ChromaDB collection purged successfully.")
        except ConnectionError:
            st.error("ChromaDB inaccessible — cannot purge.")
        except Exception as e:
            st.error(f"Purge failed: {type(e).__name__}: {e}")
            logger.exception("Purge error")
