"""Tests qualité retrieval : filtre min_score neutre + recherche ciblée par fichier.

Couvre les correctifs appliqués suite au constat de réponses faibles :
  - MIN_SCORE = 0.0 : les scores post-RRF étant normalisés en min-max, un seuil
    absolu (0.65) ne gardait que le chunk #1 et faisait échouer des questions
    pertinentes (ex. « compétences Python »).
  - Détection d'un nom de fichier dans la question → filtre ``where`` ciblé,
    y compris sur les résultats BM25 (le filtre ChromaDB ne s'applique qu'à la
    recherche vectorielle).
"""

from __future__ import annotations

import pytest

from app.indexing.vector_store import SearchResult
from app.ingestion.chunker import ChunkStrategy, TextChunk


def _search_result(doc_id: str, content: str, score: float, source: str = "src.txt", page: int = 1) -> SearchResult:
    return SearchResult(
        id=doc_id,
        content=content,
        score=score,
        metadata={"source_file": source, "page_number": page},
    )


class TestExtractDocumentFilter:
    """Détection d'un fichier nommé dans la question (app/core/rag_engine.py)."""

    def _fake_vs(self, sources: set[str]):
        class _VS:
            def get_source_files(self):
                return sources

        return _VS()

    def test_exact_filename(self):
        from app.core.rag_engine import _extract_document_filter

        vs = self._fake_vs({"CV_BENMOUH_Mohamed.pdf", "CahierDeCharge1_StageSIP_25.pdf"})
        where = _extract_document_filter("cherche dans CV_BENMOUH_Mohamed.pdf les compétences", vs)
        assert where == {"source_file": "CV_BENMOUH_Mohamed.pdf"}

    def test_case_insensitive_resolution(self):
        from app.core.rag_engine import _extract_document_filter

        vs = self._fake_vs({"CV_BENMOUH_Mohamed.pdf"})
        where = _extract_document_filter("CHERCHE DANS cv_benmouh_mohamed.pdf les compétences", vs)
        assert where == {"source_file": "CV_BENMOUH_Mohamed.pdf"}

    def test_no_filename_returns_none(self):
        from app.core.rag_engine import _extract_document_filter

        vs = self._fake_vs({"CV_BENMOUH_Mohamed.pdf"})
        assert _extract_document_filter("Quel est le CA du groupe ?", vs) is None

    def test_named_file_not_indexed_returns_none(self):
        from app.core.rag_engine import _extract_document_filter

        vs = self._fake_vs({"CV_BENMOUH_Mohamed.pdf"})
        assert _extract_document_filter("cherche dans rapport_annuel.pdf", vs) is None

    def test_filename_with_underscores(self):
        from app.core.rag_engine import _extract_document_filter

        vs = self._fake_vs({"rapport_annuel_2025.pdf"})
        where = _extract_document_filter("dis moi ce qui est dans rapport_annuel_2025.pdf", vs)
        assert where == {"source_file": "rapport_annuel_2025.pdf"}


class TestGetSourceFiles:
    """get_source_files() renvoie les fichiers distincts de la collection."""

    def test_distinct_sources(self, vector_store_fixture):
        chunks = [
            TextChunk(
                content=f"Contenu {i} du fichier {f}.",
                source_file=f,
                page_number=1,
                chunk_index=i,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            )
            for f, k in (("a.txt", 2), ("b.pdf", 3), ("a.txt", 1))
            for i in range(k)
        ]

        vector_store_fixture.initialize_collection()
        vector_store_fixture.add_chunks(chunks)

        assert vector_store_fixture.get_source_files() == {"a.txt", "b.pdf"}


class TestRetrieveWhereFiltersBm25:
    """Le filtre ``where`` (fichier nommé) s'applique aussi aux résultats BM25."""

    def test_mixed_bm25_is_restricted(self, monkeypatch):
        from app.retrieval import retriever as r

        all_results = [
            _search_result("a1", "contenu a1", 0.9, source="a.txt"),
            _search_result("a2", "contenu a2", 0.8, source="a.txt"),
            _search_result("b1", "contenu b1", 0.95, source="b.txt"),
        ]

        class _FakeVS:
            def search(self, query, top_k, where=None):
                if where:
                    return [x for x in all_results if x.metadata["source_file"] == where["source_file"]]
                return all_results

            def count(self):
                return len(all_results)

            def get_all_documents(self):
                return [
                    {"id": x.id, "text": x.content, "metadata": x.metadata}
                    for x in all_results
                ]

        monkeypatch.setattr(r, "USE_HYBRID", True)
        monkeypatch.setattr(r, "USE_HYDE", False)
        monkeypatch.setattr(r, "_get_cross_encoder", lambda: None)

        retriever = r.Retriever(_FakeVS())
        # BM25 renvoie les 3 sources mélangées ; le filtre where doit n'en garder qu'une.
        monkeypatch.setattr(
            retriever,
            "_bm25_search",
            lambda query, top_k: list(reversed(all_results)),
        )

        results = retriever.retrieve("question", top_k=6, where={"source_file": "a.txt"})
        assert results, "des résultats doivent être retournés"
        assert all(x.metadata["source_file"] == "a.txt" for x in results)

    def test_default_min_score_is_zero_keeps_all(self, monkeypatch):
        from app.retrieval import retriever as r

        class _FakeVS:
            def search(self, query, top_k, where=None):
                return [
                    _search_result("a", "contenu a", 0.2),
                    _search_result("b", "contenu b", 0.1),
                ]

        monkeypatch.setattr(r, "USE_HYBRID", False)
        monkeypatch.setattr(r, "USE_HYDE", False)
        monkeypatch.setattr(r, "_get_cross_encoder", lambda: None)

        retriever = r.Retriever(_FakeVS())
        results = retriever.retrieve("question", top_k=6)  # min_score par défaut
        assert len(results) == 2  # aucun résultat écarté par le seuil par défaut


class TestAnswerPassesDocumentFilter:
    """answer() transmet le filtre ``where`` détecté au retriever."""

    def test_answer_forwards_where(self):
        from app.core.rag_engine import RAGEngine

        engine = object.__new__(RAGEngine)
        engine.top_k = 6
        engine.min_score = 0.0

        captured: dict = {}

        class _Retriever:
            def retrieve(self, question, *, top_k, min_score, where):
                captured["where"] = where
                return []

            def format_context(self, results):
                return ""

        class _LLM:
            def generate(self, question, context):
                return "réponse"

        class _VS:
            def get_source_files(self):
                return {"CV_BENMOUH_Mohamed.pdf"}

        engine.retriever = _Retriever()
        engine.llm = _LLM()
        engine.vector_store = _VS()

        response, _ = engine.answer("cherche dans CV_BENMOUH_Mohamed.pdf les compétences")
        assert response == "réponse"
        assert captured["where"] == {"source_file": "CV_BENMOUH_Mohamed.pdf"}


class TestFollowupRewrite:
    """Réécriture des questions de suivi (app/core/rag_engine.py)."""

    def test_short_followup_is_combined(self):
        from app.core.rag_engine import build_search_query

        history = [
            {"role": "user", "content": "who is mohamed ben mouh ?"},
            {"role": "assistant", "content": "Il est élève ingénieur à l'EHTP."},
        ]
        assert build_search_query("and now ?", history) == "who is mohamed ben mouh ? and now ?"

    def test_french_connector_is_combined(self):
        from app.core.rag_engine import build_search_query

        history = [{"role": "user", "content": "cherche dans CV_BENMOUH_Mohamed.pdf les compétences"}]
        q = build_search_query("et ensuite ?", history)
        assert "CV_BENMOUH_Mohamed.pdf" in q
        assert q.endswith("et ensuite ?")

    def test_standalone_question_unchanged(self):
        from app.core.rag_engine import build_search_query

        history = [{"role": "user", "content": "who is mohamed ben mouh ?"}]
        assert build_search_query("Quel est le CA du groupe ?", history) == "Quel est le CA du groupe ?"

    def test_followup_without_history_is_unchanged(self):
        from app.core.rag_engine import build_search_query

        assert build_search_query("and now ?", []) == "and now ?"


class TestCrossEncoderLocalResolution:
    """Le reranker se charge hors-ligne via la résolution locale (comme l'embedder)."""

    def test_local_path_passed_to_cross_encoder(self, monkeypatch):
        from app.retrieval import retriever as r

        captured: dict = {}

        class _CE:
            def __init__(self, model_path, max_length=None):
                captured["path"] = model_path
                captured["max_length"] = max_length

        monkeypatch.setattr(r, "_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        monkeypatch.setattr(r, "_cross_encoder", None)
        monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)
        monkeypatch.setattr(
            "app.indexing.vector_store._resolve_local_sentence_transformers_path",
            lambda model, cache: "/app/.cache/huggingface/snapshots/x",
        )

        class _Os:
            @staticmethod
            def getenv(name, default=""):
                return "/app/.cache/huggingface"

        monkeypatch.setattr(r, "os", _Os())

        ce = r._get_cross_encoder()

        assert ce is not None
        assert captured["path"] == "/app/.cache/huggingface/snapshots/x"
        assert captured["max_length"] == 512

    def test_offline_failure_falls_back_to_none(self, monkeypatch):
        from app.retrieval import retriever as r

        class _CE:
            def __init__(self, model_path, max_length=None):
                raise RuntimeError("pas de snapshot local")

        monkeypatch.setattr(r, "_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        monkeypatch.setattr(r, "_cross_encoder", None)
        monkeypatch.setattr("sentence_transformers.CrossEncoder", _CE)

        assert r._get_cross_encoder() is None
        # sentinel : le second appel ne retente pas le chargement
        assert r._get_cross_encoder() is None
