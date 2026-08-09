"""Tests pytest pour le module Retriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.indexing.vector_store import SearchResult
from app.retrieval.retriever import Retriever


class TestRetriever:
    """Tests unitaires du retriever avec mock du vector store."""

    @pytest.fixture()
    def retriever(self):
        mock_store = MagicMock()
        return Retriever(mock_store)

    def test_retrieve_filters_by_min_score(self, retriever):
        mock_results = [
            SearchResult(id="1", content="Chunk pertinent", score=0.85, metadata={"source_file": "doc1.pdf"}),
            SearchResult(id="2", content="Chunk moyen", score=0.60, metadata={"source_file": "doc1.pdf"}),
            SearchResult(id="3", content="Chunk faible", score=0.30, metadata={"source_file": "doc2.txt"}),
        ]
        retriever.vector_store.search.return_value = mock_results

        results = retriever.retrieve("ma requete", top_k=3, min_score=0.0)

        assert len(results) == 3
        assert {r.id for r in results} == {"1", "2", "3"}

    def test_retrieve_filters_all_below_min_score(self, retriever):
        mock_results = [
            SearchResult(id="1", content="A", score=0.1, metadata={}),
            SearchResult(id="2", content="B", score=0.2, metadata={}),
        ]
        retriever.vector_store.search.return_value = mock_results

        results = retriever.retrieve("test", top_k=5, min_score=1.5)

        assert len(results) == 0

    def test_retrieve_returns_all_above_min_score(self, retriever):
        mock_results = [
            SearchResult(id="1", content="A", score=0.9, metadata={}),
            SearchResult(id="2", content="B", score=0.8, metadata={}),
        ]
        retriever.vector_store.search.return_value = mock_results

        results = retriever.retrieve("test", top_k=5, min_score=0.0)

        assert len(results) == 2

    def test_format_context_with_sources(self, retriever):
        results = [
            SearchResult(
                id="1", content="Contenu 1", score=0.9,
                metadata={"source_file": "doc.pdf", "page_number": 12},
            ),
            SearchResult(
                id="2", content="Contenu 2", score=0.8,
                metadata={"source_file": "table.sql"},
            ),
        ]

        context = retriever.format_context(results)

        assert "[Extrait 1 - doc.pdf - page 12]" in context
        assert "[Extrait 2 - table.sql]" in context
        assert "Contenu 1" in context
        assert "Contenu 2" in context

    def test_format_context_empty(self, retriever):
        context = retriever.format_context([])
        assert context == ""

    def test_format_context_no_page(self, retriever):
        results = [
            SearchResult(
                id="1", content="Texte", score=0.7,
                metadata={"source_file": "doc.txt", "page_number": -1},
            ),
        ]

        context = retriever.format_context(results)

        assert "page" not in context
        assert "Extrait 1" in context

    def test_retrieve_calls_vector_store(self, retriever):
        retriever.vector_store.search.return_value = []

        retriever.retrieve("question test", top_k=5, min_score=0.1)

        retriever.vector_store.search.assert_called_once_with(
            "question test", top_k=12, where=None
        )

    def test_retrieve_with_where_filter(self, retriever):
        retriever.vector_store.search.return_value = []

        retriever.retrieve("test", top_k=3, where={"source_file": "doc.pdf"})

        retriever.vector_store.search.assert_called_once_with(
            "test", top_k=8, where={"source_file": "doc.pdf"}
        )
