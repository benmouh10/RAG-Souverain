"""Tests pytest pour le module DocumentParser."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.parser import (
    DocumentParser,
    ExtractionMethod,
    ParsedDocument,
    ParsedPage,
    SQLParser,
    SQLRecord,
)


class TestDocumentParser:
    """Tests du parseur de documents."""

    def test_parse_directory_returns_documents(self, parsed_documents):
        assert len(parsed_documents) > 0
        assert all(isinstance(d, ParsedDocument) for d in parsed_documents)

    def test_parse_directory_filters_extensions(self, tmp_path):
        (tmp_path / "image.png").write_bytes(b"fake")
        (tmp_path / "doc.pdf").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_text("hello")

        parser = DocumentParser(ocr_enabled=False)
        docs = parser.parse_directory(tmp_path)

        extensions = {d.source_path.suffix.lower() for d in docs}
        assert ".png" not in extensions

    def test_parse_text_file(self, sample_text_file):
        parser = DocumentParser()
        doc = parser.parse_file(sample_text_file)

        assert doc.source_file == sample_text_file.name
        assert doc.page_count == 1
        assert "RAG Souverain" in doc.full_text

    def test_parse_markdown_file(self, sample_markdown_file):
        parser = DocumentParser()
        doc = parser.parse_file(sample_markdown_file)

        assert doc.source_file == sample_markdown_file.name
        assert doc.page_count == 1
        assert "Introduction" in doc.full_text

    def test_parse_nonexistent_file_raises(self):
        parser = DocumentParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/file.txt")

    def test_parse_unsupported_extension_raises(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")

        parser = DocumentParser()
        with pytest.raises(ValueError, match="Extension non supportée"):
            parser.parse_file(f)

    def test_parsed_page_is_empty(self):
        page = ParsedPage(
            content="",
            source_file="test.pdf",
            page_number=1,
            extraction_method=ExtractionMethod.PDF_NATIVE,
            file_extension=".pdf",
        )
        assert page.is_empty is True

    def test_parsed_page_not_empty(self):
        page = ParsedPage(
            content="Du texte",
            source_file="test.pdf",
            page_number=1,
            extraction_method=ExtractionMethod.PDF_NATIVE,
            file_extension=".pdf",
        )
        assert page.is_empty is False


class TestSQLParser:
    """Tests du parseur SQL."""

    def test_get_tables(self, sample_sql_db):
        parser = SQLParser(sample_sql_db)
        tables = parser.get_tables()
        parser.close()

        assert "employes" in tables
        assert "produits" in tables

    def test_parse_all_tables(self, sample_sql_db):
        parser = SQLParser(sample_sql_db)
        records = parser.parse_all_tables()
        parser.close()

        assert len(records) >= 5
        assert all(isinstance(r, SQLRecord) for r in records)

    def test_to_parsed_documents(self, sample_sql_db):
        parser = SQLParser(sample_sql_db)
        doc = parser.to_parsed_documents()
        parser.close()

        assert isinstance(doc, ParsedDocument)
        assert doc.page_count >= 5

    def test_sql_record_to_parsed_page(self, sample_sql_db):
        parser = SQLParser(sample_sql_db)
        records = parser.parse_all_tables()
        page = records[0].to_parsed_page()
        parser.close()

        assert isinstance(page, ParsedPage)
        assert page.extraction_method == ExtractionMethod.SQL
        assert "Table:" in page.content
