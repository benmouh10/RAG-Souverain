"""
Découpage adaptatif de documents parsés pour l'indexation RAG.

Stratégies (CDC) :
- Section : documents structurés (Markdown, titres numérotés)
- Fenêtre glissante : texte libre (512 tokens, overlap 50)
- Tableau : préserve les lignes de tableau intactes
- SQL : chaque enregistrement est un chunk
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

from app.ingestion.parser import ParsedDocument, ParsedPage, ExtractionMethod

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOKENIZER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

SECTION_HEADER_PATTERN = re.compile(
    r"^(?:#{1,6}\s+.+|(?:\d+(?:\.\d+)*[\.\)]\s+.+)|(?:[IVXLC]+\.\s+.+)|(?:[A-ZÀ-Ü][A-ZÀ-Ü\s\-]{2,})$)",
    re.MULTILINE,
)

# Pattern pour détecter des tableaux simples (lignes séparées par | ou avec alignements)
TABLE_PATTERN = re.compile(
    r"^(?:\|.*\|.*\||[\+\-]+[\|\-\+]+[\+\-]+)$",
    re.MULTILINE,
)


class ChunkStrategy(str, Enum):
    """Stratégie de découpage appliquée à un chunk.

    Le choix est fait par l'AdaptiveChunker selon le type de contenu :
    sections pour les documents structurés, fenêtre glissante pour le texte
    libre, table pour les tableaux markdown et SQL pour les enregistrements.
    """

    SECTION = "section"
    SLIDING_WINDOW = "sliding_window"
    TABLE = "table"
    SQL = "sql"


@dataclass(frozen=True)
class TextChunk:
    """Chunk textuel prêt pour l'embedding, avec métadonnées de traçabilité."""

    content: str
    source_file: str
    page_number: int | None
    chunk_index: int
    strategy: ChunkStrategy
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        """Métadonnées sérialisables pour ChromaDB."""
        base = {
            "source_file": self.source_file,
            "page_number": self.page_number if self.page_number is not None else -1,
            "chunk_index": self.chunk_index,
            "strategy": self.strategy.value,
            "version": self.version,
        }
        base.update(self.metadata)
        return base


class AdaptiveChunker:
    """Découpe adaptative : sections pour documents structurés, fenêtre fixe sinon."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap doit être strictement inférieur à chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer_model = tokenizer_model

    def chunk_documents(self, documents: list[ParsedDocument]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        global_index = 0
        for document in documents:
            doc_chunks = self.chunk_document(document)
            for chunk in doc_chunks:
                chunks.append(
                    TextChunk(
                        content=chunk.content,
                        source_file=chunk.source_file,
                        page_number=chunk.page_number,
                        chunk_index=global_index,
                        strategy=chunk.strategy,
                        metadata=chunk.metadata,
                    )
                )
                global_index += 1
        return chunks

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        local_index = 0

        is_markdown = document.source_path.suffix.lower() == ".md"

        if is_markdown and document.pages:
            full_text = document.full_text
            if self._is_structured(full_text):
                chunks.extend(
                    self._chunk_by_sections(
                        full_text,
                        source_file=document.source_file,
                        page_number=None,
                        start_index=local_index,
                    )
                )
                return chunks

        for page in document.pages:
            if page.is_empty:
                continue
            page_chunks = self._chunk_page(page, local_index)
            chunks.extend(page_chunks)
            local_index += len(page_chunks)

        return chunks

    def _chunk_page(self, page: ParsedPage, start_index: int) -> list[TextChunk]:
        # Check if it's SQL first
        if page.extraction_method == ExtractionMethod.SQL:
            return self._chunk_sql(page, start_index)
        
        # Check if it's a table
        if self._is_table(page.content):
            return self._chunk_by_table(page, start_index)
        
        # Otherwise, structured or sliding window
        if self._is_structured(page.content):
            return self._chunk_by_sections(
                page.content,
                source_file=page.source_file,
                page_number=page.page_number,
                start_index=start_index,
            )
        return self._chunk_by_sliding_window(
            page.content,
            source_file=page.source_file,
            page_number=page.page_number,
            start_index=start_index,
        )

    def _is_table(self, text: str) -> bool:
        lines = text.strip().split("\n")
        table_lines = [line for line in lines if TABLE_PATTERN.match(line)]
        return len(table_lines) >= 2  # At least header + separator + one row?

    def _chunk_by_table(self, page: ParsedPage, start_index: int) -> list[TextChunk]:
        lines = page.content.split("\n")
        if not lines:
            return []
        
        chunks: list[TextChunk] = []
        index = start_index
        
        # Identify header and separator
        header = ""
        separator = ""
        current_table = []
        
        for i, line in enumerate(lines):
            if TABLE_PATTERN.match(line):
                if not header:
                    header = line
                elif not separator and (set(line.strip()) <= {"+", "-", "|"}):
                    separator = line
                else:
                    current_table.append(line)
            else:
                # Flush current table if we have content
                if current_table:
                    chunk_content = "\n".join([header, separator] + current_table)
                    chunks.append(
                        TextChunk(
                            content=chunk_content,
                            source_file=page.source_file,
                            page_number=page.page_number,
                            chunk_index=index,
                            strategy=ChunkStrategy.TABLE,
                            metadata={"has_header": bool(header), "has_separator": bool(separator)},
                        )
                    )
                    index += 1
                    header = ""
                    separator = ""
                    current_table = []
        
        # Flush remaining
        if current_table:
            chunk_content = "\n".join([header, separator] + current_table)
            chunks.append(
                TextChunk(
                    content=chunk_content,
                    source_file=page.source_file,
                    page_number=page.page_number,
                    chunk_index=index,
                    strategy=ChunkStrategy.TABLE,
                    metadata={"has_header": bool(header), "has_separator": bool(separator)},
                )
            )
        
        # If no table lines found, use sliding window
        if not chunks:
            return self._chunk_by_sliding_window(
                page.content,
                source_file=page.source_file,
                page_number=page.page_number,
                start_index=start_index,
            )
        
        return chunks

    def _chunk_sql(self, page: ParsedPage, start_index: int) -> list[TextChunk]:
        """Pour SQL, chaque page/enregistrement est un chunk complet."""
        return [
            TextChunk(
                content=page.content,
                source_file=page.source_file,
                page_number=page.page_number,
                chunk_index=start_index,
                strategy=ChunkStrategy.SQL,
                metadata=page.metadata,
            )
        ]

    def _is_structured(self, text: str) -> bool:
        headers = SECTION_HEADER_PATTERN.findall(text)
        return len(headers) >= 2

    def _chunk_by_sections(
        self,
        text: str,
        *,
        source_file: str,
        page_number: int | None,
        start_index: int,
    ) -> list[TextChunk]:
        sections = self._split_into_sections(text)
        chunks: list[TextChunk] = []
        index = start_index

        for title, body in sections:
            section_text = f"{title}\n{body}".strip() if body else title.strip()
            if not section_text:
                continue

            token_count = self._count_tokens(section_text)
            if token_count <= self.chunk_size:
                chunks.append(
                    TextChunk(
                        content=section_text,
                        source_file=source_file,
                        page_number=page_number,
                        chunk_index=index,
                        strategy=ChunkStrategy.SECTION,
                        metadata={"section_title": title.strip()},
                    )
                )
                index += 1
            else:
                sub_chunks = self._chunk_by_sliding_window(
                    section_text,
                    source_file=source_file,
                    page_number=page_number,
                    start_index=index,
                    section_title=title.strip(),
                )
                chunks.extend(sub_chunks)
                index += len(sub_chunks)

        return chunks

    def _chunk_by_sliding_window(
        self,
        text: str,
        *,
        source_file: str,
        page_number: int | None,
        start_index: int,
        section_title: str | None = None,
    ) -> list[TextChunk]:
        token_ids = self._encode(text)
        if not token_ids:
            return []

        stride = self.chunk_size - self.chunk_overlap
        chunks: list[TextChunk] = []
        index = start_index

        for start in range(0, len(token_ids), stride):
            end = start + self.chunk_size
            window_ids = token_ids[start:end]
            if not window_ids:
                break

            content = self._decode(window_ids).strip()
            if not content:
                continue

            metadata: dict[str, Any] = {}
            if section_title:
                metadata["section_title"] = section_title

            chunks.append(
                TextChunk(
                    content=content,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=index,
                    strategy=ChunkStrategy.SLIDING_WINDOW,
                    metadata=metadata,
                )
            )
            index += 1

            if end >= len(token_ids):
                break

        return chunks

    @staticmethod
    def _split_into_sections(text: str) -> list[tuple[str, str]]:
        matches = list(SECTION_HEADER_PATTERN.finditer(text))
        if not matches:
            return [("", text.strip())]

        sections: list[tuple[str, str]] = []

        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                sections.append(("Préambule", preamble))

        for i, match in enumerate(matches):
            title = match.group(0).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            sections.append((title, body))

        return sections

    @lru_cache(maxsize=1)
    def _get_tokenizer(self):
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(self.tokenizer_model)
        except ImportError as exc:
            raise ImportError(
                "transformers requis pour le comptage de tokens. "
                "Installez sentence-transformers."
            ) from exc

    def _encode(self, text: str) -> list[int]:
        tokenizer = self._get_tokenizer()
        return tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids: list[int]) -> str:
        tokenizer = self._get_tokenizer()
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def _count_tokens(self, text: str) -> int:
        return len(self._encode(text))
