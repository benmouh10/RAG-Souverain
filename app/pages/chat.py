"""Chat RAG page — interactive assistant with intent guard and clean sources."""

from __future__ import annotations

import logging
import time

import streamlit as st

from app.app_ui import (
    GREETING_RESPONSE,
    _load_database,
    _load_llm_class,
    _load_observability,
    build_chat_message,
    build_source_payloads,
    compute_cosine_score,
    dedupe_sources,
    init_rag_engine,
    is_greeting,
    truncate_snippet,
)
from app.core.rag_engine import _extract_document_filter, build_search_query
from app.core.sql_query_engine import is_structured_query

logger = logging.getLogger("rag_ui")

SNIPPET_MAX_CHARS = 250


def _render_source_drawer(sources_list: list[dict]) -> None:
    """Render deduplicated sources with truncated, expandable snippets."""
    sources_list = dedupe_sources(sources_list)
    with st.expander("Sources and similarity scores", expanded=False):
        for i, source in enumerate(sources_list, start=1):
            sf = source.get("source_file") or "Unknown source"
            pg = source.get("page_number") or "-"
            pg = pg if str(pg).strip() and str(pg).strip() != "None" else "-"
            sc = source.get("score") or 0.0
            sc = sc if sc not in (None, "None", "none") else 0.0
            ver = source.get("version") or "-"
            ver = ver if str(ver).strip() and str(ver).strip() != "None" else "-"
            ct = source.get("content") or ""
            ct = ct if ct.strip() and ct.strip() != "None" else ""

            preview = truncate_snippet(ct, max_chars=SNIPPET_MAX_CHARS)
            truncated = len(preview) < len(ct.strip())

            header = (
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<div style='font-weight:700'>{i}. {sf} "
                f"<span style='font-weight:400;margin-left:6px'>page {pg} v{ver}</span></div>"
                f"<div class='source-score'>score: {sc:.4f}</div></div>"
                f"<span class='source-pill'>Indexed source</span>"
            )
            if truncated:
                details = (
                    f"<details style='margin-top:8px'>"
                    f"<summary class='source-readmore'>Show full text ({(len(ct))} chars)</summary>"
                    f"<div class='source-snippet'>{ct}</div>"
                    f"</details>"
                )
                st.markdown(
                    f"<div class='source-card'>{header}"
                    f"<div class='source-snippet'>{preview}…</div>{details}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='source-card'>{header}"
                    f"<div class='source-snippet'>{ct}</div></div>",
                    unsafe_allow_html=True,
                )


def _render_citations(citations_list: list) -> None:
    if not isinstance(citations_list, list) or not citations_list:
        return
    with st.expander("Citations (exact quotes)", expanded=False):
        for j, cit in enumerate(citations_list, start=1):
            if not isinstance(cit, dict):
                continue
            quote = cit.get("quote") or ""
            chunk_id = cit.get("chunk_id") or ""
            if quote:
                chunk_badge = f' <span style="color:#666;font-size:11px">(chunk: {chunk_id})</span>' if chunk_id else ''
                quote_html = f"<br/><em>&ldquo;{quote}&rdquo;</em>"
                st.markdown(
                    f"<div style='background:#fef9c3;border-left:4px solid #eab308;padding:8px 12px;margin:4px 0;border-radius:4px'>"
                    f"<strong>Citation {j}</strong>{chunk_badge}{quote_html}</div>",
                    unsafe_allow_html=True,
                )


def render_chat_page() -> None:
    rag = init_rag_engine()
    user_id = st.session_state.get("user_id")

    Database = _load_database()
    QueryLogger = _load_observability()
    db = Database()
    query_logger = QueryLogger()

    if "cleanup_done" not in st.session_state:
        deleted = db.cleanup_old_messages(max_age_days=30)
        if deleted:
            logger.info("Auto-cleaned %d old chat messages", deleted)
        deleted_none = db.cleanup_none_messages(user_id)
        if deleted_none:
            logger.info("Auto-cleaned %d 'None' chat messages", deleted_none)
        st.session_state["cleanup_done"] = True

    active_conv_id = st.session_state.get("active_conversation_id") or 1

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Chat RAG")
    st.caption("Query your local documents in a simple and secure way.")
    st.markdown("</div>", unsafe_allow_html=True)

    for msg in st.session_state["messages"]:
        role = msg.get("role") or "user"
        raw_content = msg.get("content") or ""
        content = raw_content if raw_content.strip() and raw_content.strip() != "None" else ""
        with st.chat_message(role):
            st.markdown(f"<div class='chat-{role}'>{content}</div>", unsafe_allow_html=True)
            if role == "assistant" and msg.get("sources"):
                sources_list = msg["sources"]
                if isinstance(sources_list, list) and sources_list:
                    cleaned = [
                        s for s in sources_list
                        if isinstance(s, dict)
                        and s.get("source_file")
                        and str(s.get("source_file")).strip()
                        and str(s.get("source_file")).strip().lower() not in ("none", "unknown source")
                    ]
                    if cleaned:
                        _render_source_drawer(cleaned)
            if role == "assistant" and msg.get("citations"):
                _render_citations(msg["citations"])

    # Mo2 / UC_Export — export de la conversation en PDF (rôle User et Admin).
    if st.session_state["messages"]:
        from app.app_ui import export_chat_to_pdf

        pdf_bytes = export_chat_to_pdf(
            st.session_state["messages"],
            st.session_state.get("username", "user"),
        )
        st.download_button(
            "Export Conversation (PDF)",
            data=pdf_bytes,
            file_name="sovereign_rag_conversation.pdf",
            mime="application/pdf",
            key="export_pdf_button",
        )

    # Search & Retrieval controls expander
    try:
        source_files = sorted(list(rag.vector_store.get_source_files()))
    except Exception:
        source_files = []

    with st.expander("🔍 Search & Retrieval Controls", expanded=False):
        c1, c2, c3 = st.columns([2, 1, 1])
        selected_file_filter = c1.selectbox(
            "Target Document",
            options=["All Documents"] + source_files,
            index=0,
            key="chat_file_filter_select",
        )
        chat_top_k = c2.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=10,
            value=st.session_state.get("top_k", 4),
            key="chat_top_k_slider",
        )
        toggle_rerank = c3.checkbox(
            "CrossEncoder Reranker",
            value=st.session_state.get("toggle_reranker", True),
            key="toggle_reranker",
        )

    user_query = st.chat_input("Ask your question to the sovereign RAG...")
    if user_query:
        # Auto-name conversation on first user query if titled "Nouvelle conversation"
        current_convs = db.list_conversations(user_id)
        active_conv = next((c for c in current_convs if c["conv_id"] == active_conv_id), None)
        if active_conv and active_conv.get("title") in ("Nouvelle conversation", "New conversation"):
            auto_title = user_query.strip()[:35]
            db.rename_conversation(active_conv_id, auto_title)

        st.session_state["messages"].append(build_chat_message("user", user_query))
        st.session_state["chat_history"] = list(st.session_state["messages"])
        db.save_chat_message(user_id, "user", user_query, conversation_id=active_conv_id)

        query_start_time = time.time()

        # Historique de conversation (hors message courant) — sert à la
        # réécriture des questions de suivi, à la génération, et à la
        # détection d'intention structurée (chemin text-to-SQL).
        conv_history = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in st.session_state["messages"][:-1]
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]

        # Suivi conversationnel : « and now ? » devient une requête autonome
        # adossée à la dernière question ; un fichier nommé dans la question
        # déclenche une recherche ciblée (filtre ``where``) ou sélectionne la
        # base SQLite à interroger côté text-to-SQL.
        search_query = build_search_query(user_query, conv_history)
        db_file_hint = selected_file_filter if selected_file_filter != "All Documents" else None

        handled = False
        full_response = ""
        source_payloads: list[dict] = []
        citations: list[dict] = []

        # Intent guard: short conversational greetings skip retrieval entirely.
        if is_greeting(user_query):
            full_response = GREETING_RESPONSE
            handled = True
        elif rag.llm is not None and rag.llm.circuit_open:
            # M2 — circuit breaker ouvert : réponse gracieuse immédiate, sans
            # passer par le retriever ni le LLM (le trafic réseau est inutile).
            full_response = (
                "Ollama is temporarily unavailable (too many recent failures). "
                "Check that the service is running, and try again."
            )
            st.warning(full_response)
            handled = True
        elif rag.should_attempt_sql(search_query):
            # Chemin text-to-SQL : la question ressemble à une agrégation ou
            # à un filtre structuré (« combien », « average », « top 5 »...)
            # et au moins une base SQLite a été ingérée. On génère une
            # requête SQL depuis le vrai schéma, on l'exécute en lecture
            # seule, et on renvoie le résultat exact — sans repasser par le
            # LLM pour reformuler les chiffres. Tout échec (génération,
            # validation, exécution) retombe silencieusement sur le
            # pipeline RAG classique ci-dessous.
            with st.status("1. Detecting a structured/aggregate question...", expanded=True) as status:
                status.update(label="2. Generating SQL from the database schema...", state="running")
                sql_result = rag.try_sql_answer(search_query, file_hint=db_file_hint)
                if sql_result is not None:
                    status.update(label="3. Executing against the SQLite database...", state="running")
                    full_response = sql_result.answer_html
                    source_payloads = [{
                        "content": f"SQL executed on {sql_result.db_file}:\n{sql_result.sql}",
                        "score": 1.0,
                        "source_file": sql_result.db_file,
                        "page_number": "SQL",
                    }]
                    status.update(label="Response ready (SQL query).", state="complete")
                    handled = True
                else:
                    status.update(
                        label="Structured query path unavailable — falling back to document search...",
                        state="error",
                    )

        if not handled:
            where = _extract_document_filter(search_query, rag.vector_store)
            if where is None and selected_file_filter != "All Documents":
                where = {"source_file": selected_file_filter}

            with st.status("1. Querying ChromaDB...", expanded=True) as status:
                try:
                    results = rag.retriever.retrieve(search_query, top_k=chat_top_k, where=where)
                    formatted_context = rag.retriever.format_context(results)
                except ConnectionError:
                    full_response = "ChromaDB is inaccessible. Make sure the service is running."
                    st.error(full_response)
                    results = []
                    formatted_context = ""
                except Exception as e:
                    full_response = f"Search error: {type(e).__name__}: {e}"
                    st.error(full_response)
                    logger.exception("Retrieval error for query: %s", user_query[:80])
                    results = []
                    formatted_context = ""

                status.update(label="2. Generating via Mistral...", state="running")

                response_area = st.empty()
                full_response = ""
                if rag.llm is None:
                    full_response = "LLM unavailable — Ollama not found. Start Ollama with `ollama serve`."
                    response_area.markdown(f"<div class='chat-assistant'>{full_response}</div>", unsafe_allow_html=True)
                else:
                    try:
                        for chunk in rag.llm.generate_stream(user_query, formatted_context, conversation_history=conv_history):
                            full_response += chunk
                            response_area.markdown(f"<div class='chat-assistant'>{full_response}</div>", unsafe_allow_html=True)
                    except ConnectionError:
                        full_response = "Lost connection to Ollama during generation."
                        response_area.markdown(f"<div class='chat-assistant'>{full_response}</div>", unsafe_allow_html=True)
                    except TimeoutError:
                        full_response = "Generation timed out (>120s). The model may be overloaded."
                        response_area.markdown(f"<div class='chat-assistant'>{full_response}</div>", unsafe_allow_html=True)
                    except Exception as exc:
                        full_response = f"Generation error: {type(exc).__name__}"
                        response_area.markdown(f"<div class='chat-assistant'>{full_response}</div>", unsafe_allow_html=True)
                        logger.exception("LLM generation error")

                source_payloads = build_source_payloads(results)

                OllamaLLM = _load_llm_class()
                parsed = OllamaLLM.parse_structured_response(full_response)
                citations = parsed.get("citations", [])
                if citations:
                    full_response = parsed["answer"]

                status.update(label="Response ready.", state="complete")

        st.session_state["messages"].append(
            build_chat_message("assistant", full_response, sources=source_payloads, citations=citations)
        )
        st.session_state["chat_history"] = list(st.session_state["messages"])
        db.save_chat_message(user_id, "assistant", full_response, sources=source_payloads, conversation_id=active_conv_id)

        try:
            top1_score = compute_cosine_score(rag, user_query)
            mean_score = sum(s.get("score", 0.0) for s in source_payloads) / len(source_payloads) if source_payloads else 0.0
            source_files = list({s.get("source_file", "") for s in source_payloads if s.get("source_file")})
            total_latency = (time.time() - query_start_time) * 1000
            query_logger.log(
                question=user_query,
                top1_score=top1_score,
                mean_score=mean_score,
                num_chunks=len(source_payloads),
                llm_latency_ms=0.0,
                total_latency_ms=total_latency,
                model_used=getattr(rag.llm, "model_name", "") if rag.llm else "",
                source_files=source_files,
            )
        except Exception as e:
            logger.debug("Observability log failed: %s", e)

        st.rerun()
