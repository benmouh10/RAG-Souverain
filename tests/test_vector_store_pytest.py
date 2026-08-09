"""Tests pytest pour le module ChromaVectorStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.indexing.vector_store import ChromaVectorStore, SearchResult
from app.ingestion.chunker import TextChunk, ChunkStrategy


class TestChromaVectorStore:
    """Tests du vector store ChromaDB (avec répertoire temporaire)."""

    def test_initialize_collection(self, vector_store_fixture):
        collection = vector_store_fixture.initialize_collection()

        assert collection is not None
        assert vector_store_fixture.count() == 0

    def test_add_chunks_and_count(self, vector_store_fixture):
        chunks = [
            TextChunk(
                content="Premier chunk de test avec du contenu suffisant.",
                source_file="test.txt",
                page_number=1,
                chunk_index=i,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            )
            for i in range(5)
        ]

        vector_store_fixture.initialize_collection()
        added = vector_store_fixture.add_chunks(chunks)

        assert added == 5
        assert vector_store_fixture.count() == 5

    def test_search_returns_results(self, vector_store_fixture):
        chunks = [
            TextChunk(
                content="Le RAG Souverain est un système local.",
                source_file="doc.pdf",
                page_number=1,
                chunk_index=0,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            ),
            TextChunk(
                content="La souveraineté des données est essentielle.",
                source_file="doc.pdf",
                page_number=2,
                chunk_index=1,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            ),
        ]

        vector_store_fixture.initialize_collection()
        vector_store_fixture.add_chunks(chunks)

        results = vector_store_fixture.search("souveraineté", top_k=2)

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score > 0

    def test_search_scores_are_valid(self, vector_store_fixture):
        chunks = [
            TextChunk(
                content="Test de score de similarité cosinus.",
                source_file="test.txt",
                page_number=1,
                chunk_index=0,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            ),
        ]

        vector_store_fixture.initialize_collection()
        vector_store_fixture.add_chunks(chunks)

        results = vector_store_fixture.search("test", top_k=1)

        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0 + 1e-6

    def test_clear_removes_all_chunks(self, vector_store_fixture):
        chunks = [
            TextChunk(
                content="Chunk à supprimer.",
                source_file="test.txt",
                page_number=1,
                chunk_index=0,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            ),
        ]

        vector_store_fixture.initialize_collection()
        vector_store_fixture.add_chunks(chunks)
        assert vector_store_fixture.count() > 0

        remaining = vector_store_fixture.clear()
        assert remaining == 0

    def test_empty_search(self, vector_store_fixture):
        vector_store_fixture.initialize_collection()
        results = vector_store_fixture.search("anything", top_k=5)

        assert len(results) == 0

    def test_reset_recreates_collection(self, vector_store_fixture):
        vector_store_fixture.initialize_collection()
        chunks = [
            TextChunk(
                content="Chunk temporaire.",
                source_file="test.txt",
                page_number=1,
                chunk_index=0,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            ),
        ]
        vector_store_fixture.add_chunks(chunks)
        assert vector_store_fixture.count() > 0

        vector_store_fixture.initialize_collection(reset=True)
        assert vector_store_fixture.count() == 0

    def test_persist_and_reload(self, tmp_chroma_dir):
        from tests.conftest import MockEmbeddingFunction

        chunks = [
            TextChunk(
                content="Données persistantes pour test.",
                source_file="test.txt",
                page_number=1,
                chunk_index=0,
                strategy=ChunkStrategy.SLIDING_WINDOW,
            ),
        ]

        mock_ef = MockEmbeddingFunction()
        vs1 = ChromaVectorStore(persist_directory=tmp_chroma_dir, embedding_function=mock_ef)
        vs1.initialize_collection()
        vs1.add_chunks(chunks)

        vs2 = ChromaVectorStore(persist_directory=tmp_chroma_dir, embedding_function=mock_ef)
        vs2.initialize_collection()
        assert vs2.count() == 1
