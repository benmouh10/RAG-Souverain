"""System Parameters page — admin-only model & retrieval configuration."""

from __future__ import annotations

import streamlit as st

from app.app_ui import init_rag_engine


def render_system_parameters_page() -> None:
    rag = init_rag_engine()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='page-title'>System Parameters</div>", unsafe_allow_html=True)
    st.caption("Tune generation and retrieval settings. Values persist for the session.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='params-layout'>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        with st.container():
            st.markdown("<div class='param-card'>", unsafe_allow_html=True)
            st.markdown("<div class='param-title'>LLM generation</div>", unsafe_allow_html=True)
            model_name = st.text_input(
                "Model name",
                value=st.session_state.get("llm_model", "mistral"),
                disabled=True,
                help="Model is fixed to the locally served LLM (Mistral).",
            )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("llm_temperature", 0.1)),
                step=0.01,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='param-card'>", unsafe_allow_html=True)
            st.markdown("<div class='param-title'>Indexing</div>", unsafe_allow_html=True)
            chunk_size = st.slider(
                "Chunk size (tokens)",
                min_value=128,
                max_value=1024,
                value=int(st.session_state.get("chunk_size", 512)),
                step=64,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        with st.container():
            st.markdown("<div class='param-card'>", unsafe_allow_html=True)
            st.markdown("<div class='param-title'>Retrieval</div>", unsafe_allow_html=True)
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=12,
                value=int(
                    st.session_state.get("top_k", rag.top_k if hasattr(rag, "top_k") else 4)
                ),
                step=1,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Apply configuration", use_container_width=True):
        st.session_state["llm_model"] = model_name.strip() or "mistral"
        st.session_state["llm_temperature"] = temperature
        st.session_state["top_k"] = top_k
        st.session_state["chunk_size"] = chunk_size

        if rag.llm is not None:
            rag.llm.temperature = temperature
        if hasattr(rag, "top_k"):
            rag.top_k = top_k

        if model_name.strip() and model_name.strip() != getattr(rag.llm, "model_name", None):
            st.session_state.pop("rag_engine", None)

        st.success("Configuration saved and applied.")

    st.markdown("---")
    st.caption(
        "Note: new documents are indexed with the selected chunk size. "
        "Changing the model reinitializes the engine on the next page load."
    )
