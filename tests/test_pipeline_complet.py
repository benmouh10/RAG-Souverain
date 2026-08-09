"""
Script de validation du pipeline complet (Parser -> Chunker -> VectorStore).
"""

from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.parser import DocumentParser, SQLParser  # noqa: E402
from app.ingestion.chunker import AdaptiveChunker  # noqa: E402
from app.indexing.vector_store import ChromaVectorStore, SearchResult  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
TEST_DB_PATH = DATA_DIR / "test_pipeline_database.db"
TEST_CHROMA_DIR = DATA_DIR / "chroma_test"
TEST_CHROMA_COLLECTION = "rag_documents_test"
SEPARATOR = "=" * 72

@pytest.mark.integration


def create_test_database() -> None:
    """Crée une base SQL dédiée aux tests, isolée des données de production."""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()
    
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE employes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            prenom TEXT NOT NULL,
            age INTEGER,
            poste TEXT,
            date_embauche TEXT
        )
    """)

    employes_data = [
        ("Dupont", "Jean", 32, "Développeur", "2022-01-15"),
        ("Martin", "Sophie", 28, "Designer", "2023-03-20"),
        ("Durand", "Paul", 40, "Chef de projet", "2020-06-01"),
    ]
    cursor.executemany("""
        INSERT INTO employes (nom, prenom, age, poste, date_embauche)
        VALUES (?, ?, ?, ?, ?)
    """, employes_data)

    cursor.execute("""
        CREATE TABLE produits (
            code TEXT PRIMARY KEY,
            nom TEXT NOT NULL,
            prix REAL,
            stock INTEGER
        )
    """)

    produits_data = [
        ("P001", "Ordinateur portable", 899.99, 10),
        ("P002", "Souris sans fil", 29.99, 150),
        ("P003", "Clavier mécanique", 129.99, 50),
    ]
    cursor.executemany("""
        INSERT INTO produits (code, nom, prix, stock)
        VALUES (?, ?, ?, ?)
    """, produits_data)

    conn.commit()
    conn.close()


def print_search_results(results: list[SearchResult], query: str) -> None:
    print(f"\nRecherche pour: \"{query}\"")
    print(f"Résultats trouvés: {len(results)}")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {result.score:.4f}) ---")
        print(f"Source: {result.metadata.get('source_file')}, Page: {result.metadata.get('page_number')}, Strategy: {result.metadata.get('strategy')}")
        preview = result.content[:400]
        suffix = "..." if len(result.content) > 400 else ""
        print(f"{preview}{suffix}")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    _configure_stdout()
    DATA_DIR.mkdir(exist_ok=True)
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    print("=== TEST PIPELINE COMPLET ===")
    print()

    # Step 1: Prepare test data
    print("1. Préparation des données de test...")
    create_test_database()
    print()

    # Step 2: Parse all documents
    print("2. Parsing des documents...")
    parser = DocumentParser()
    documents = parser.parse_directory(DOCUMENTS_DIR)
    
    # Add SQL data
    sql_parser = SQLParser(TEST_DB_PATH)
    sql_doc = sql_parser.to_parsed_documents()
    documents.append(sql_doc)
    
    print(f"   Total documents parsés: {len(documents)}")
    print()

    # Step 3: Chunk all documents
    print("3. Chunking des documents...")
    chunker = AdaptiveChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"   Total chunks: {len(chunks)}")
    print()

    # Step 4: Index in vector store
    print("4. Indexation dans ChromaDB...")
    chroma_path = os.getenv("CHROMA_TEST_PERSIST_DIR", str(TEST_CHROMA_DIR))
    chroma_collection = os.getenv("CHROMA_TEST_COLLECTION", TEST_CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(
        persist_directory=chroma_path,
        collection_name=chroma_collection,
    )
    vector_store.initialize_collection(reset=True)
    added = vector_store.add_chunks(chunks)
    print(f"   Chunks indexés: {added}")
    print()

    # Step 5: Test search
    print("5. Tests de recherche...")
    test_queries = [
        "Jean Dupont développeur",
        "Ordinateur portable prix",
        "Tableau Nom Âge Ville",
        "Fichier texte UTF-8",
    ]
    
    for query in test_queries:
        results = vector_store.search(query, top_k=3)
        print_search_results(results, query)
    
    sql_parser.close()

    print(f"\n{SEPARATOR}")
    print("Validation du pipeline complet terminée avec succès.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
