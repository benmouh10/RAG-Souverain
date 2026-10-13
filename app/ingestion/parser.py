"""
Module d'ingestion et parsing de documents pour RAG Souverain.

Supporte PDF natifs (pdfplumber / PyMuPDF), OCR Tesseract pour PDF scannés,
fichiers texte (.txt, .md), et bases de données SQL (SQLite).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".sql"}
DEFAULT_OCR_LANGUAGE = "fra"
DEFAULT_MIN_TEXT_LENGTH = 10
DEFAULT_OCR_DPI = 300


class ExtractionMethod(str, Enum):
    """Méthode utilisée pour extraire le texte d'un document.

    Permet à la chaîne RAG d'adapter le découpage (ex. : SQL → un chunk par
    enregistrement) et de tracer la provenance de chaque chunk.
    """

    PDF_NATIVE = "pdf_native"
    PDF_OCR = "pdf_ocr"
    TEXT = "text"
    SQL = "sql"


@dataclass(frozen=True)
class ParsedPage:
    """Fragment de document avec métadonnées de provenance."""

    content: str
    source_file: str
    page_number: int | None
    extraction_method: ExtractionMethod
    file_extension: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.content.strip()


@dataclass
class ParsedDocument:
    """Document parsé regroupant une ou plusieurs pages."""

    source_path: Path
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def source_file(self) -> str:
        return self.source_path.name

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.content for page in self.pages if page.content.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)


@dataclass(frozen=True)
class SQLRecord:
    """Enregistrement SQL avec métadonnées structurées."""

    table_name: str
    record_id: Any
    content: str
    metadata: dict[str, Any]
    db_path: Path

    def to_parsed_page(self) -> ParsedPage:
        return ParsedPage(
            content=self.content,
            source_file=self.db_path.name,
            page_number=None,
            extraction_method=ExtractionMethod.SQL,
            file_extension=".sql",
            metadata=self.metadata,
        )


class SQLParser:
    """Parseur SQL pour l'extraction de données depuis SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        import sqlite3

        self.db_path = Path(db_path)
        self._connection: sqlite3.Connection | None = None

    def _connect(self):
        import sqlite3

        if not self._connection:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    def get_tables(self) -> list[str]:
        """Récupère la liste des tables de la base de données.

        Exclut les tables internes de SQLite (``sqlite_sequence``, etc.) qui
        n'ont aucune valeur métier et pollueraient l'index RAG.
        """
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        return [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

    def get_primary_key(self, table_name: str) -> str | None:
        """Récupère la clé primaire d'une table."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        for row in cursor.fetchall():
            if row[5]:  # pk column
                return row[1]
        return None

    def parse_table(self, table_name: str) -> list[SQLRecord]:
        """Extrait tous les enregistrements d'une table et les formate."""
        import json

        conn = self._connect()
        primary_key = self.get_primary_key(table_name)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        records: list[SQLRecord] = []
        for row in rows:
            row_dict = dict(row)
            record_id = row_dict.get(primary_key, len(records)) if primary_key else len(records)

            content_lines = [
                f"[Fichier DB/SQL: {self.db_path.name}]",
                f"Table: {table_name}",
                f"Enregistrement ID: {record_id}",
                "Données :",
            ]
            for key, value in row_dict.items():
                if value is not None and str(value).strip() != "":
                    content_lines.append(f"  - {key}: {value}")

            records.append(
                SQLRecord(
                    table_name=table_name,
                    record_id=record_id,
                    content="\n".join(content_lines),
                    metadata={
                        "source_file": self.db_path.name,
                        "table_name": table_name,
                        "record_id": record_id,
                        "columns": list(row_dict.keys()),
                        "data": row_dict,
                    },
                    db_path=self.db_path,
                )
            )

        return records

    def parse_all_tables(self) -> list[SQLRecord]:
        """Parse toutes les tables de la base de données."""
        all_records: list[SQLRecord] = []
        tables = self.get_tables()
        for table in tables:
            all_records.extend(self.parse_table(table))
        return all_records

    def to_parsed_documents(self) -> ParsedDocument:
        """Convertit tous les enregistrements SQL en un ParsedDocument."""
        all_records = self.parse_all_tables()
        pages = [record.to_parsed_page() for record in all_records]
        return ParsedDocument(
            source_path=Path(self.db_path),
            pages=pages,
        )


class DocumentParser:
    """Parse un répertoire ou des fichiers individuels en documents structurés."""

    def __init__(
        self,
        *,
        ocr_enabled: bool = True,
        ocr_language: str = DEFAULT_OCR_LANGUAGE,
        min_text_length: int = DEFAULT_MIN_TEXT_LENGTH,
        ocr_dpi: int = DEFAULT_OCR_DPI,
        recursive: bool = True,
    ) -> None:
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        self.min_text_length = min_text_length
        self.ocr_dpi = ocr_dpi
        self.recursive = recursive

    def parse_directory(self, directory: str | Path) -> list[ParsedDocument]:
        """Lit tous les fichiers supportés d'un répertoire."""
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Répertoire introuvable : {root}")

        pattern = "**/*" if self.recursive else "*"
        files = sorted(
            path
            for path in root.glob(pattern)
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        documents: list[ParsedDocument] = []
        for file_path in files:
            try:
                documents.append(self.parse_file(file_path))
            except Exception:
                logger.exception("Échec du parsing de %s", file_path)
        return documents

    def parse_file(self, file_path: str | Path) -> ParsedDocument:
        """Parse un fichier unique selon son extension."""
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extension non supportée : {suffix}. "
                f"Extensions acceptées : {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        if suffix == ".pdf":
            pages = self._parse_pdf(path)
        else:
            pages = [self._parse_text_file(path)]

        return ParsedDocument(source_path=path, pages=pages)

    def iter_pages(self, directory: str | Path) -> Iterator[ParsedPage]:
        """Itère sur toutes les pages de tous les documents d'un répertoire."""
        for document in self.parse_directory(directory):
            yield from document.pages

    def _parse_pdf(self, path: Path) -> list[ParsedPage]:
        """Extrait le texte natif page par page, avec fallback OCR si nécessaire."""
        native_pages = self._extract_pdf_native(path)

        if not self.ocr_enabled:
            return native_pages

        pages: list[ParsedPage] = []
        for page in native_pages:
            if self._needs_ocr(page.content):
                ocr_page = self._extract_pdf_page_ocr(path, page.page_number)
                pages.append(ocr_page)
            else:
                pages.append(page)
        return pages

    def _extract_pdf_native(self, path: Path) -> list[ParsedPage]:
        """Extraction native via pdfplumber, repli PyMuPDF."""
        try:
            return self._extract_with_pdfplumber(path)
        except ImportError:
            logger.warning("pdfplumber indisponible, bascule vers PyMuPDF.")
        except Exception:
            logger.exception("pdfplumber a échoué pour %s, bascule vers PyMuPDF.", path)

        return self._extract_with_pymupdf(path)

    def _extract_with_pdfplumber(self, path: Path) -> list[ParsedPage]:
        import pdfplumber

        pages: list[ParsedPage] = []
        with pdfplumber.open(path) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append(
                    ParsedPage(
                        content=text.strip(),
                        source_file=path.name,
                        page_number=index,
                        extraction_method=ExtractionMethod.PDF_NATIVE,
                        file_extension=path.suffix.lower(),
                    )
                )
        return pages

    def _extract_with_pymupdf(self, path: Path) -> list[ParsedPage]:
        import fitz

        pages: list[ParsedPage] = []
        with fitz.open(path) as doc:
            for index in range(doc.page_count):
                page = doc.load_page(index)
                text = page.get_text("text") or ""
                pages.append(
                    ParsedPage(
                        content=text.strip(),
                        source_file=path.name,
                        page_number=index + 1,
                        extraction_method=ExtractionMethod.PDF_NATIVE,
                        file_extension=path.suffix.lower(),
                    )
                )
        return pages

    def _extract_pdf_page_ocr(self, path: Path, page_number: int | None) -> ParsedPage:
        """OCR d'une page PDF via PyMuPDF (rendu) + pytesseract."""
        if page_number is None:
            return ParsedPage(
                content="",
                source_file=path.name,
                page_number=None,
                extraction_method=ExtractionMethod.PDF_OCR,
                file_extension=path.suffix.lower(),
            )

        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "OCR requis : installez pymupdf, pytesseract et pillow."
            ) from exc

        with fitz.open(path) as doc:
            if page_number < 1 or page_number > doc.page_count:
                raise ValueError(f"Numéro de page invalide : {page_number}")

            page = doc.load_page(page_number - 1)
            matrix = fitz.Matrix(self.ocr_dpi / 72, self.ocr_dpi / 72)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            text = pytesseract.image_to_string(image, lang=self.ocr_language)

        return ParsedPage(
            content=text.strip(),
            source_file=path.name,
            page_number=page_number,
            extraction_method=ExtractionMethod.PDF_OCR,
            file_extension=path.suffix.lower(),
        )

    def _parse_text_file(self, path: Path) -> ParsedPage:
        """Lit un fichier .txt ou .md en UTF-8 avec repli latin-1."""
        content = self._read_text_with_fallback(path)
        return ParsedPage(
            content=content.strip(),
            source_file=path.name,
            page_number=None,
            extraction_method=ExtractionMethod.TEXT,
            file_extension=path.suffix.lower(),
        )

    @staticmethod
    def _read_text_with_fallback(path: Path) -> str:
        """Read a text file robustly.

        Attempts UTF-8, UTF-8-sig, then Latin-1. If all fail (or any
        exception is raised), decode the raw bytes with ``errors='replace'``
        so the caller never receives a ``UnicodeDecodeError``.
        """
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return path.read_text(encoding=encoding)
            except Exception:
                continue
        return path.read_bytes().decode("utf-8", errors="replace")


    def _needs_ocr(self, text: str) -> bool:
        return len(text.strip()) < self.min_text_length
