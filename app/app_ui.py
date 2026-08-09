"""Shared helpers for the Streamlit UI (RAG Souverain).

Kept importable as ``app.app_ui`` for backwards compatibility. The actual
page rendering lives under :mod:`app.pages` and the entry point is
:mod:`app.streamlit_app`.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("rag_ui")

if TYPE_CHECKING:
    from app.evaluation.eval_pipeline import RAGEngine


# ---------------------------------------------------------------------------
# Lazy imports (fallback for out-of-package execution)
# ---------------------------------------------------------------------------
def _load_rag_engine_class() -> Any:
    try:
        from app.evaluation.eval_pipeline import RAGEngine
    except ModuleNotFoundError:
        from evaluation.eval_pipeline import RAGEngine
    return RAGEngine


def _load_eval_helpers() -> tuple[Any, Any]:
    try:
        from app.evaluation.eval_pipeline import export_results, run_evaluation
    except ModuleNotFoundError:
        from evaluation.eval_pipeline import export_results, run_evaluation
    return export_results, run_evaluation


def _load_llm_class() -> Any:
    try:
        from app.generation.llm import OllamaLLM
    except ModuleNotFoundError:
        from generation.llm import OllamaLLM
    return OllamaLLM


def _load_ingestion_helpers() -> tuple[Any, Any]:
    try:
        from app.ingestion.chunker import AdaptiveChunker
        from app.ingestion.parser import DocumentParser
    except ModuleNotFoundError:
        from ingestion.chunker import AdaptiveChunker
        from ingestion.parser import DocumentParser
    return AdaptiveChunker, DocumentParser


def _load_database() -> Any:
    try:
        from app.database import Database
    except ModuleNotFoundError:
        from database import Database
    return Database


def _load_observability() -> Any:
    try:
        from app.observability import QueryLogger
    except ModuleNotFoundError:
        from observability import QueryLogger
    return QueryLogger


# ---------------------------------------------------------------------------
# Business errors
# ---------------------------------------------------------------------------
class IngestError(Exception):
    """Error during file ingestion."""
    pass


class OllamaConnectionError(Exception):
    """Ollama unreachable."""
    pass


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def init_user_session(user_id: int) -> None:
    """Reset session data for a new user."""
    key = "_last_user_id"
    last = st.session_state.get(key)
    if last != user_id:
        st.session_state[key] = user_id
        st.session_state["messages"] = []
        st.session_state["chat_history"] = []
        st.session_state.pop("rag_engine", None)
        st.session_state.pop("history_loaded", None)
        st.session_state.pop("cleanup_done", None)


def build_chat_message(role: str, content: str, sources: list[dict[str, Any]] | None = None, citations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    safe = content if content and content.strip() and content.strip() != "None" else ""
    payload: dict[str, Any] = {"role": role, "content": safe}
    if sources is not None:
        payload["sources"] = sources
    if citations is not None:
        payload["citations"] = citations
    return payload


def get_upload_widget_key() -> int:
    return st.session_state.get("uploader_id", 0)


def reset_upload_widget_state() -> int:
    st.session_state["uploader_id"] = st.session_state.get("uploader_id", 0) + 1
    st.session_state.pop("uploaded_file_name", None)
    try:
        st.rerun()
    except Exception:
        pass
    return st.session_state["uploader_id"]


def build_source_payloads(results: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for result in results:
        meta = getattr(result, "metadata", None) or {}
        sf = meta.get("source_file") or "Unknown source"
        sf = sf if sf and str(sf).strip() and str(sf).strip().lower() not in ("none",) else "Unknown source"
        pg = meta.get("page_number")
        pg = pg if pg is not None and str(pg).strip() and str(pg).strip().lower() not in ("none",) else -1
        payloads.append(
            {
                "content": result.content or "",
                "score": getattr(result, "score", 0.0) or 0.0,
                "source_file": sf,
                "page_number": pg,
                "chunk_index": meta.get("chunk_index") or -1,
            }
        )
    return payloads


def _sanitize_for_pdf(text: str) -> str:
    """Replace Unicode characters unsupported by Helvetica in fpdf2."""
    if not text:
        return ""
    replacements = {
        "\u2018": "'", "\u2019": "'",   # guillemets simples
        "\u201c": '"', "\u201d": '"',   # guillemets doubles
        "\u2013": "-", "\u2014": "-",   # tirets
        "\u2026": "...",                 # ellipsis
        "\u0153": "oe", "\u0152": "OE", # ligatures
        "\u00a0": " ",                   # espace insécable
        "\uf0b7": "*", "\u2022": "*",   # puces
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Supprimer tout caractère restant hors latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_chat_to_pdf(messages: list[dict[str, Any]], username: str) -> bytes:
    """Export chat history to PDF via fpdf2."""
    from fpdf import FPDF

    class ChatPDF(FPDF):
        """Sous-classe fpdf2 : entête (titre + utilisateur) et pied de page
        avec pagination ``Page X / Y`` pour l'export de conversation."""

        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 10, "RAG Souverain - Conversation History", new_x="LMARGIN", new_y="NEXT", align="C")
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 6, f"User: {username}", new_x="LMARGIN", new_y="NEXT", align="C")
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    pdf = ChatPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for msg in messages:
        role = msg.get("role", "user")
        content = _sanitize_for_pdf(msg.get("content") or "")
        label = "You" if role == "user" else "Assistant"

        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(230, 240, 255) if role == "user" else pdf.set_fill_color(230, 255, 240)
        pdf.cell(0, 7, f"  {label}", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5, content)
        pdf.ln(2)

        if role == "assistant" and msg.get("sources"):
            pdf.set_font("Helvetica", "I", 8)
            for i, src in enumerate(msg["sources"], 1):
                src_file = _sanitize_for_pdf(src.get("source_file") or "?")
                score = src.get("score") or 0
                pdf.cell(0, 4, f"    Source {i}: {src_file} (score: {score:.4f})", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    return pdf.output()


# ---------------------------------------------------------------------------
# RAG Engine initialization (with typed errors)
# ---------------------------------------------------------------------------
def init_rag_engine(force: bool = False, model_name: str | None = None) -> Any:
    rag_engine_cls = _load_rag_engine_class()
    user_id = st.session_state.get("user_id")
    engine_key = f"rag_engine_{user_id}"

    if engine_key not in st.session_state or force:
        kwargs: dict[str, Any] = {}
        if model_name:
            kwargs["model_name"] = model_name
        elif st.session_state.get("llm_model"):
            kwargs["model_name"] = st.session_state["llm_model"]
        if user_id is not None:
            kwargs["user_id"] = user_id
        try:
            st.session_state[engine_key] = rag_engine_cls(**kwargs)
        except ConnectionError as e:
            st.error(f"Cannot reach Ollama: {e}")
            st.stop()
        except Exception as e:
            st.error(f"RAG engine initialization error: {e}")
            logger.exception("RAG engine init failed")
            st.stop()
    return st.session_state[engine_key]


def get_chroma_count(rag: Any) -> int:
    try:
        return rag.vector_store.count()
    except ConnectionError:
        logger.warning("ChromaDB inaccessible during count")
        return 0
    except Exception as e:
        logger.warning("ChromaDB count error: %s", e)
        return 0


def delete_file_from_vector_store(filename: str, rag: Any | None = None) -> int:
    """Delete every vector chunk belonging to ``filename`` from the shared store.

    Returns the number of chunks removed (0 if the file is not present).
    """
    active_rag = rag or init_rag_engine()
    return active_rag.vector_store.delete_by_source(filename)


def aggregate_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge per-version source rows into one row per distinct file."""
    files: dict[str, dict[str, Any]] = {}
    for s in sources:
        name = s.get("source_file", "unknown")
        row = files.setdefault(
            name,
            {"source_file": name, "chunk_count": 0, "ingested_at": s.get("ingested_at", "")},
        )
        row["chunk_count"] += int(s.get("chunk_count", 0))
        if not row["ingested_at"]:
            row["ingested_at"] = s.get("ingested_at", "")
    return sorted(files.values(), key=lambda d: d["source_file"].lower())


def compute_cosine_score(rag: Any, query: str) -> float:
    """Real floating-point cosine similarity for the top-1 vector match."""
    try:
        raw = rag.vector_store.search(query, top_k=1)
        if raw:
            return float(raw[0].score)
    except Exception as e:
        logger.debug("Cosine score computation failed: %s", e)
    return 0.0


# ---------------------------------------------------------------------------
# Ingestion with typed errors
# ---------------------------------------------------------------------------
def ingest_uploaded_file(uploaded_file, rag: Any | None = None) -> tuple[int, str]:
    active_rag = rag or init_rag_engine()
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getvalue())

    suffix = file_path.suffix.lower()

    try:
        if suffix == ".db":
            added = active_rag.ingest_sql(file_path)
            return added, file_path.name

        AdaptiveChunker, DocumentParser = _load_ingestion_helpers()
        parser = DocumentParser()
        parsed_documents = parser.parse_file(file_path)

        if parsed_documents.page_count == 0:
            raise IngestError(f"File '{file_path.name}' contains no usable content.")

        chunk_size = st.session_state.get("chunk_size", 512)
        chunker = AdaptiveChunker(chunk_size=chunk_size)
        chunks = chunker.chunk_documents([parsed_documents])

        if not chunks:
            raise IngestError(f"No chunks generated from '{file_path.name}'. Check the file format.")

        added = active_rag.vector_store.add_chunks(chunks)
        active_rag.vector_store.initialize_collection()
        return added, file_path.name

    except IngestError:
        raise
    except FileNotFoundError:
        raise IngestError(f"File not found after upload: {file_path.name}")
    except ValueError as e:
        raise IngestError(f"Unsupported format or corrupted file: {e}")
    except ConnectionError:
        raise OllamaConnectionError("ChromaDB inaccessible — cannot index.")
    except Exception as e:
        logger.exception("Unexpected error during indexing of %s", file_path.name)
        raise IngestError(f"Internal indexing error: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------
_GREETING_PATTERNS = re.compile(
    r"^(hey|hello|hi|hiya|yo|salut|bonjour|bonsoir|bonsoir|coucou|hola|slt|hello there|hi there)$",
    re.IGNORECASE,
)

GREETING_RESPONSE = "Hello! How can I assist you with your indexed documents today?"


def is_greeting(query: str) -> bool:
    """True if the query is a short conversational greeting.

    Used to short-circuit the vector search so casual greetings do not pull
    random indexed chunks.
    """
    q = (query or "").strip().strip("!?.,; ")
    if not q:
        return False
    # Born length check: a real greeting is short; longer input (even starting
    # with "hello") is treated as a real search intent, not a greeting.
    if len(q) > 12:
        return False
    return bool(_GREETING_PATTERNS.match(q))


def dedupe_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate sources by identical content (or content + page/file)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for src in sources:
        content = (src.get("content") or "").strip()
        file_name = src.get("source_file") or ""
        page = src.get("page_number") or -1
        key = f"{file_name}|{page}|{content}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(src)
    return unique


def truncate_snippet(content: str, max_chars: int = 250) -> str:
    """Truncate a source snippet to ``max_chars`` characters."""
    if not content:
        return ""
    content = content.strip()
    if len(content) <= max_chars:
        return content
    ellipsis = "…"
    budget = max(1, max_chars - len(ellipsis))
    cut = content[:budget]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + ellipsis


# Backward-compatible entry point (now routed through app.streamlit_app).
def main() -> None:
    """Wrapper de compatibilité : redirige vers ``app.streamlit_app.main``."""
    from app.streamlit_app import main as _streamlit_main

    _streamlit_main()


if __name__ == "__main__":
    main()
