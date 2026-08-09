"""Fixtures partagées pour tous les tests pytest du RAG Souverain."""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Dimension du modèle paraphrase-multilingual-MiniLM-L12-v2
_EMBEDDING_DIM = 384


class MockEmbeddingFunction:
    """Fonction d'embedding factice pour les tests — pas de modèle HuggingFace requis."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        rng = np.random.default_rng(seed=42)
        return [rng.random(_EMBEDDING_DIM).tolist() for _ in input]

    def embed_query(self, query: str) -> list[float]:
        rng = np.random.default_rng(seed=42)
        return rng.random(_EMBEDDING_DIM).tolist()

    def name(self) -> str:
        return "mock-embedding"


@pytest.fixture()
def tmp_chroma_dir(tmp_path: Path) -> Path:
    """Répertoire ChromaDB temporaire isolé pour chaque test."""
    return tmp_path / "chroma_test"


@pytest.fixture()
def sample_text_file(tmp_path: Path) -> Path:
    """Crée un fichier texte de test temporaire."""
    f = tmp_path / "test_doc.txt"
    f.write_text(
        "Ceci est un document de test pour le RAG Souverain.\n"
        "Il contient plusieurs lignes de texte.\n"
        "Caractères spéciaux français : éèàôûçñ\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def sample_markdown_file(tmp_path: Path) -> Path:
    """Crée un fichier Markdown de test temporaire."""
    f = tmp_path / "test_doc.md"
    f.write_text(
        "# Introduction\n"
        "Ceci est un document Markdown de test.\n\n"
        "## Section 1\n"
        "Contenu de la section 1.\n\n"
        "## Section 2\n"
        "Contenu de la section 2.\n",
        encoding="utf-8",
    )
    return f


@pytest.fixture()
def sample_sql_db(tmp_path: Path) -> Path:
    """Crée une base SQLite de test avec des données."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE employes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            prenom TEXT NOT NULL,
            age INTEGER,
            poste TEXT
        )
    """)
    cursor.executemany(
        "INSERT INTO employes (nom, prenom, age, poste) VALUES (?, ?, ?, ?)",
        [
            ("Dupont", "Jean", 32, "Développeur"),
            ("Martin", "Sophie", 28, "Designer"),
            ("Durand", "Paul", 40, "Chef de projet"),
        ],
    )

    cursor.execute("""
        CREATE TABLE produits (
            code TEXT PRIMARY KEY,
            nom TEXT NOT NULL,
            prix REAL,
            stock INTEGER
        )
    """)
    cursor.executemany(
        "INSERT INTO produits (code, nom, prix, stock) VALUES (?, ?, ?, ?)",
        [
            ("P001", "Ordinateur portable", 899.99, 10),
            ("P002", "Souris sans fil", 29.99, 150),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def parsed_documents():
    """Parse les vrais documents du projet pour les tests d'intégration."""
    from app.ingestion.parser import DocumentParser

    parser = DocumentParser(ocr_enabled=False)
    docs = parser.parse_directory(DOCUMENTS_DIR)
    return docs


@pytest.fixture()
def vector_store_fixture(tmp_chroma_dir):
    """Crée un ChromaVectorStore temporaire avec un embedding mock (pas de HuggingFace)."""
    from app.indexing.vector_store import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=tmp_chroma_dir,
        embedding_function=MockEmbeddingFunction(),
    )



