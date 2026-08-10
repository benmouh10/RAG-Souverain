"""
Script de test de bout-en-bout du systeme RAG.
Valide l'orchestration entre le Retriever et le modele LLM local (Ollama).
"""

import os
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.indexing.vector_store import ChromaVectorStore
from app.retrieval.retriever import Retriever
from app.generation.llm import OllamaLLM

DATA_DIR = PROJECT_ROOT / "data"
SEPARATOR = "=" * 72

@pytest.mark.integration
def test_rag_end_to_end():
    """Test d'intégration Retrieval + LLM local (Ollama).

    Exclu du CI rapide (pytest.ini). Se skippe proprement si la collection
    vectorielle est vide (exécuter d'abord ``test_pipeline_complet``) ou si
    Ollama est indisponible. Valide qu'une question reçoit une réponse non
    vide, avec le retriever (min_score faible pour tolérer un petit corpus).
    """
    chroma_path = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
    chroma_collection = os.getenv("CHROMA_COLLECTION", "rag_documents")
    vector_store = ChromaVectorStore(
        persist_directory=chroma_path,
        collection_name=chroma_collection,
    )

    if vector_store.count() == 0:
        pytest.skip("Collection vectorielle vide — exécuter d'abord tests/test_pipeline_complet.py")

    retriever = Retriever(vector_store)
    llm = OllamaLLM(model_name="mistral")

    if not llm.health_check(timeout=5):
        pytest.skip("Ollama indisponible (host.docker.internal:11434 ou localhost:11434)")

    results = retriever.retrieve(
        "Quel est le poste de Jean Dupont ?",
        top_k=3,
        min_score=0.2,
    )
    formatted_context = retriever.format_context(results)

    response = llm.generate(
        "Quel est le poste de Jean Dupont ?",
        formatted_context,
    )

    assert response and response.strip(), "Le LLM a retourné une réponse vide."
    assert "Ollama" not in response, f"Échec de génération : {response[:200]}"


def main():
    print(SEPARATOR)
    print("=== TEST RAG COMPLET (Retrieval + Ollama LLM) ===")
    print(SEPARATOR)

    # 1. Initialisation de la base vectorielle
    print("1. Connexion au Vector Store...")
    chroma_path = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
    chroma_collection = os.getenv("CHROMA_COLLECTION", "rag_documents")
    vector_store = ChromaVectorStore(
        persist_directory=chroma_path,
        collection_name=chroma_collection,
    )
    
    if vector_store.count() == 0:
        print("ATTENTION : La collection de test est vide.")
        print("Veuillez d'abord exécuter 'python tests/test_pipeline_complet.py' pour l'indexer.")
        return

    print(f"   -> Base connectée ({vector_store.count()} chunks disponibles).\n")

    # 2. Initialisation des composants RAG
    print("2. Initialisation du Retriever et du client Ollama...")
    retriever = Retriever(vector_store)
    llm = OllamaLLM(model_name="mistral") # Mistral est le modèle par défaut du CDC
    print("   -> Composants prêts.\n")

    # 3. Test de requêtes
    queries = [
        "Quel est le poste de Jean Dupont ?",
        "Combien coûte l'ordinateur portable et combien en a-t-on en stock ?",
        "Quelle est la capitale de l'Australie ?" # Question hors contexte pour tester l'anti-hallucination
    ]

    for i, query in enumerate(queries, start=1):
        print(SEPARATOR)
        print(f"QUESTION {i} : {query}")
        print("-" * 30)
        
        # A. Retrieval
        print("[*] Recherche de contexte...")
        start_time = time.time()
        results = retriever.retrieve(query, top_k=3, min_score=0.2)
        formatted_context = retriever.format_context(results)
        print(f"    {len(results)} chunks trouvés en {time.time() - start_time:.2f}s.")
        
        # B. Génération
        print("[*] Génération de la réponse (Streaming) :")
        print("    >> ", end="", flush=True)
        
        try:
            for chunk in llm.generate_stream(query, formatted_context):
                print(chunk, end="", flush=True)
            print() # Nouvelle ligne à la fin
        except Exception as e:
            print(f"\n[Erreur de génération] : {e}")
            
        print("\n[*] Sources utilisées :")
        if not results:
            print("    Aucune source pertinente trouvée.")
        for res in results:
            source = res.metadata.get('source_file', 'Inconnu')
            page = res.metadata.get('page_number')
            page_str = f", page {page}" if page and page != -1 else ""
            print(f"    - {source}{page_str} (Score: {res.score:.2f})")
        print("\n")


if __name__ == "__main__":
    # Correction d'encodage pour l'affichage console sous Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
