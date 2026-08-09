"""Tests pytest pour le module AdaptiveChunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import (
    AdaptiveChunker,
    ChunkStrategy,
    TextChunk,
)
from app.ingestion.parser import DocumentParser, ExtractionMethod, ParsedPage, ParsedDocument


class TestAdaptiveChunker:
    """Tests du chunker adaptatif."""

    def test_chunk_documents_returns_chunks(self, parsed_documents):
        chunker = AdaptiveChunker()
        chunks = chunker.chunk_documents(parsed_documents)

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_chunks_have_global_index(self, parsed_documents):
        chunker = AdaptiveChunker()
        chunks = chunker.chunk_documents(parsed_documents)

        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_sliding_window_strategy(self):
        text = "Mot " * 200  # ~200 mots
        page = ParsedPage(
            content=text,
            source_file="test.txt",
            page_number=1,
            extraction_method=ExtractionMethod.TEXT,
            file_extension=".txt",
        )
        doc = ParsedDocument(source_path=Path("test.txt"), pages=[page])

        chunker = AdaptiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        assert all(c.strategy == ChunkStrategy.SLIDING_WINDOW for c in chunks)

    def test_section_strategy(self):
        text = (
            "# Titre 1\n"
            "Contenu de la section 1 avec assez de texte pour remplir.\n\n"
            "# Titre 2\n"
            "Contenu de la section 2 avec assez de texte pour remplir.\n\n"
            "# Titre 3\n"
            "Contenu de la section 3 avec assez de texte pour remplir.\n"
        )
        page = ParsedPage(
            content=text,
            source_file="test.md",
            page_number=1,
            extraction_method=ExtractionMethod.TEXT,
            file_extension=".md",
        )
        doc = ParsedDocument(source_path=Path("test.md"), pages=[page])

        chunker = AdaptiveChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 1

    def test_sql_strategy(self):
        page = ParsedPage(
            content="--- Table: employes ---\nID: 1\nnom: Dupont\nprenom: Jean",
            source_file="employes",
            page_number=None,
            extraction_method=ExtractionMethod.SQL,
            file_extension=".sql",
            metadata={"table_name": "employes", "record_id": 1},
        )
        doc = ParsedDocument(source_path=Path("test.db"), pages=[page])

        chunker = AdaptiveChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].strategy == ChunkStrategy.SQL

    def test_empty_page_skipped(self):
        page = ParsedPage(
            content="",
            source_file="empty.pdf",
            page_number=1,
            extraction_method=ExtractionMethod.PDF_NATIVE,
            file_extension=".pdf",
        )
        doc = ParsedDocument(source_path=Path("empty.pdf"), pages=[page])

        chunker = AdaptiveChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_to_metadata(self):
        chunk = TextChunk(
            content="Test content",
            source_file="doc.pdf",
            page_number=3,
            chunk_index=5,
            strategy=ChunkStrategy.SLIDING_WINDOW,
            metadata={"extra": "data"},
        )
        meta = chunk.to_metadata()

        assert meta["source_file"] == "doc.pdf"
        assert meta["page_number"] == 3
        assert meta["chunk_index"] == 5
        assert meta["strategy"] == "sliding_window"
        assert meta["extra"] == "data"

    def test_chunk_overlap_validation(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            AdaptiveChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="chunk_overlap"):
            AdaptiveChunker(chunk_size=100, chunk_overlap=150)

    def test_no_chunks_for_empty_document(self):
        doc = ParsedDocument(source_path=Path("empty.pdf"), pages=[])
        chunker = AdaptiveChunker()
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0
