"""
Moteur RAG — couche cœur de l'application
==========================================

Le ``RAGEngine`` orchestre la chaîne *retrieval + génération* en s'appuyant
sur les briques du projet : ``ChromaVectorStore`` (indexation vectorielle),
``Retriever`` (recherche) et ``OllamaLLM`` (génération locale 100 % souveraine).

Historique du remaniement :
- C2 : le moteur a été extrait de ``app.evaluation.eval_pipeline`` (dont le
  rôle se limite désormais à l'évaluation RAGAS) vers ce module ``app.core``,
  afin de respecter la séparation des responsabilités annoncée dans la
  conception : le moteur RAG est un composant de production consommé par
  l'interface, pas une dépendance du module d'évaluation.
- C1/Mo4 : le versionnage des documents est **global** (clé = ``filename``),
  indépendamment de l'utilisateur qui déclenche l'indexation. ``doc_versions``
  ne contient plus de colonne ``user_id`` ; la version d'un fichier est donc
  partagée par tous les administrateurs.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import Any

from app.core.sql_query_engine import (
    SQLAnswer,
    SQLGenerationError,
    SQLSafetyError,
    SQLQueryEngine,
    is_structured_query,
)
from app.generation.llm import OllamaLLM
from app.indexing.vector_store import ChromaVectorStore
from app.retrieval.retriever import Retriever

logger = logging.getLogger("rag_engine")

# ---------------------------------------------------------------------------
# Configuration (variables d'environnement pour Docker)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_ollama_base_url() -> str:
    from_env = os.getenv("OLLAMA_BASE_URL")
    if from_env:
        return from_env
    if os.getenv("IN_DOCKER") == "1" or Path("/.dockerenv").exists():
        return "http://host.docker.internal:11434"
    return "http://localhost:11434"


OLLAMA_BASE_URL = _resolve_ollama_base_url()
GENERATOR_MODEL = "mistral"  # Modèle utilisé pour générer les réponses
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "data" / "chroma")))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_documents")
TOP_K = 6
# Le seuil est désactivé (0.0) : après la fusion RRF, les scores sont normalisés
# en min-max (le meilleur résultat vaut toujours 1.0, les suivants s'écrasent à
# ~0.05-0.15). Un seuil absolu comme 0.65 ne gardait que le chunk #1 et faisait
# échouer des questions pertinentes (ex. « compétences Python »). Le tri final
# repose sur top_k + rerank ; min_score reste un garde-fou optionnel.
MIN_SCORE = 0.0

# Regex de détection d'un nom de fichier explicite dans la question
# (ex. « cherche dans CV_BENMOUH_Mohamed.pdf »). Le nom est ensuite résolu
# contre les fichiers réellement indexés avant d'appliquer un filtre ``where``.
# Convention du dépôt : noms de fichiers sans espace (séparateur ``_``).
_FILENAME_RE = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_.\-]*\.(?:pdf|md|txt|sql|db))", re.IGNORECASE)


def _extract_document_filter(question: str, vector_store: ChromaVectorStore) -> dict[str, Any] | None:
    """Détecte un nom de fichier dans la question et le résout vers l'index.

    Retourne ``{"source_file": <nom indexé>}`` si un fichier nommé dans la
    question existe dans la collection (comparaison insensible à la casse),
    sinon ``None`` (recherche globale).
    """
    indexed = vector_store.get_source_files()
    for match in _FILENAME_RE.finditer(question):
        asked = match.group(1)
        for name in indexed:
            if name.lower() == asked.lower():
                return {"source_file": name}
    return None


# ---------------------------------------------------------------------------
# Suivi conversationnel : réécriture des questions de suivi en requête autonome
# ---------------------------------------------------------------------------
_FOLLOWUP_PREFIXES = (
    "and", "and now", "and then", "then", "what about",
    "et", "et maintenant", "et puis", "ensuite", "alors", "puis", "quoi",
)


def _is_followup(question: str) -> bool:
    """Une question est un suivi si elle est très courte ou commence par un
    connecteur de continuation (« and now ? », « et ensuite ? ») — elle n'a
    pas de sujet autonome pour le retrieval."""
    q = question.strip().lower()
    if not q:
        return False
    tokens = re.findall(r"\w+", q)
    if len(tokens) <= 3:
        return True
    return q.startswith(_FOLLOWUP_PREFIXES)


def build_search_query(question: str, history: list[dict[str, str]]) -> str:
    """Transforme une question de suivi en requête de recherche autonome.

    Ex. « and now ? » après « who is mohamed ben mouh ? » devient
    « who is mohamed ben mouh ? and now ? » : le retrieval retrouve le même
    document (et le filtre ``where`` détecte le nom de fichier s'il en
    figurait un). Les questions autonomes ne sont pas modifiées.
    """
    q = question.strip()
    if not _is_followup(q):
        return q
    for msg in reversed(history or []):
        if msg.get("role") == "user" and msg.get("content"):
            return f"{msg['content']} {q}".strip()
    return q


class RAGEngine:
    """
    Moteur RAG complet qui enchaîne retrieval + génération.
    Utilise les composants existants du projet.
    """

    def __init__(
        self,
        chroma_persist_dir: str | Path = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION,
        model_name: str = GENERATOR_MODEL,
        top_k: int = TOP_K,
        min_score: float = MIN_SCORE,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.model_name = model_name

        # Global shared collection: all roles (admin & users) query the same
        # knowledge base. No user-level collection scoping.
        scoped_collection = collection_name

        self.ollama_available = self._check_ollama()

        # Initialisation du Vector Store
        logger.info("Initialisation du Vector Store (persist_dir=%s, collection=%s)...",
                    chroma_persist_dir, scoped_collection)
        self.vector_store = ChromaVectorStore(
            persist_directory=chroma_persist_dir,
            collection_name=scoped_collection,
        )
        self.vector_store.initialize_collection()

        # Initialisation du LLM de génération (si Ollama disponible)
        if self.ollama_available:
            self.llm = OllamaLLM(model_name=model_name)
        else:
            self.llm = None

        # Initialisation du Retriever
        self.retriever = Retriever(vector_store=self.vector_store, llm=self.llm)

        # Moteur text-to-SQL : chemin alternatif pour les questions
        # structurées/agrégées (voir sql_query_engine module docstring).
        # Toujours instancié (découverte de fichiers .db + introspection ne
        # nécessitent pas Ollama) ; seule la génération de SQL a besoin du
        # LLM, vérifié au moment de l'appel dans try_sql_answer().
        self.sql_engine = SQLQueryEngine()

        # DB pour le versionnage global des documents indexés (C1/Mo4)
        from app.database import Database

        self._db = Database()

        logger.info("RAGEngine prêt. Collection: %d docs, Modèle: %s, Ollama: %s",
                    self.vector_store.count(), model_name,
                    "✓" if self.ollama_available else "✗")

    def _check_ollama(self) -> bool:
        """Vérifie si Ollama est accessible."""
        import requests
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=15)
            return resp.status_code == 200
        except Exception:
            return False

    def ingest_sql(self, db_path: str | Path) -> int:
        """Parse une base SQLite et indexe tous les enregistrements.

        Comme ``ingest_document``, l'ingestion est idempotente : un fichier
        ``.db`` dont le contenu n'a pas changé (même hash) n'est pas
        ré-indexé, et un fichier modifié voit ses anciens chunks supprimés
        avant l'ajout des nouveaux — sans ce nettoyage, ré-uploader deux fois
        le même fichier (ou une version mise à jour) empilait indéfiniment
        des chunks dupliqués/obsolètes dans ChromaDB.
        """
        from app.database import compute_file_hash
        from app.ingestion.chunker import AdaptiveChunker
        from app.ingestion.parser import SQLParser

        db_path = Path(db_path)
        filename = db_path.name
        file_hash = compute_file_hash(db_path.read_bytes())

        existing_hash = self._db.get_file_hash(filename)
        if existing_hash == file_hash:
            logger.info("SQL '%s' inchangé depuis le dernier import — ignoré.", filename)
            return 0

        parser = SQLParser(db_path)
        sql_doc = parser.to_parsed_documents()
        parser.close()

        chunker = AdaptiveChunker()
        chunks = chunker.chunk_documents([sql_doc])

        # Supprime les chunks d'une éventuelle ingestion précédente du même
        # fichier (par nom) avant d'ajouter les nouveaux, pour éviter les
        # doublons/contenus obsolètes.
        removed = self.vector_store.delete_by_source(filename)
        if removed:
            logger.info("%d ancien(s) chunk(s) supprimé(s) pour '%s' avant ré-indexation.", removed, filename)

        doc_version = self._db.record_ingestion(filename, file_hash, len(chunks))
        chunks = [dataclasses.replace(chunk, version=doc_version.version) for chunk in chunks]

        added = self.vector_store.add_chunks(chunks)
        logger.info("SQL ingéré : %d chunks depuis %s (version %d)", added, db_path, doc_version.version)
        return added

    def ingest_document(
        self,
        file_path: str | Path,
        filename: str,
        file_data: bytes,
        *,
        chunk_size: int | None = None,
    ) -> tuple[int, int]:
        """Indexe un document avec suivi de version global (C1/Mo4).

        Returns:
            (chunks_added, version_number)
        """
        from app.database import compute_file_hash
        from app.ingestion.chunker import AdaptiveChunker
        from app.ingestion.parser import DocumentParser

        file_hash = compute_file_hash(file_data)

        existing_hash = self._db.get_file_hash(filename)
        if existing_hash == file_hash:
            latest = self._db.get_latest_version(filename)
            return 0, latest.version if latest else 1

        parser = DocumentParser()
        parsed_documents = parser.parse_file(file_path)

        if parsed_documents.page_count == 0:
            raise ValueError(f"Le fichier '{filename}' ne contient aucun contenu exploitable.")

        chunker = AdaptiveChunker(chunk_size=chunk_size) if chunk_size else AdaptiveChunker()
        chunks = chunker.chunk_documents([parsed_documents])

        if not chunks:
            raise ValueError(f"Aucun chunk généré depuis '{filename}'.")

        # Supprime les chunks d'une éventuelle ingestion précédente du même
        # fichier (première indexation "sale" ou nouvelle version) avant
        # d'ajouter les nouveaux chunks. Sans cela, ré-uploader une nouvelle
        # version d'un fichier empile ses chunks à côté des anciens dans
        # ChromaDB : les deux versions restent alors mélangées au retrieval
        # (aucun filtre par version n'est appliqué en recherche normale),
        # produisant des réponses incohérentes ou contradictoires.
        removed = self.vector_store.delete_by_source(filename)
        if removed:
            logger.info("%d ancien(s) chunk(s) supprimé(s) pour '%s' avant ré-indexation.", removed, filename)

        # La version n'est enregistrée qu'une fois le parsing/chunking réussi,
        # ce qui évite les lignes orphelines en cas d'échec (chunk_count = 0).
        doc_version = self._db.record_ingestion(filename, file_hash, len(chunks))
        version = doc_version.version

        # TextChunk est un dataclass frozen (immuable) : `chunk.version = ...`
        # lève FrozenInstanceError. dataclasses.replace() construit de
        # nouvelles instances versionnées à la place.
        chunks = [dataclasses.replace(chunk, version=version) for chunk in chunks]

        added = self.vector_store.add_chunks(chunks)
        self.vector_store.initialize_collection()

        logger.info("Document ingéré : %d chunks, version %d, fichier=%s", added, version, filename)
        return added, version

    def should_attempt_sql(self, question: str) -> bool:
        """True si la question a l'air structurée/agrégée ET qu'au moins une
        base SQLite a été ingérée. Ne garantit pas le succès de la
        génération/exécution — seulement que la tentative vaut le coût."""
        return (
            self.llm is not None
            and not self.llm.circuit_open
            and is_structured_query(question)
            and self.sql_engine.has_databases()
        )

    def try_sql_answer(self, question: str, file_hint: str | None = None) -> SQLAnswer | None:
        """Tente le chemin text-to-SQL. Retourne ``None`` (jamais ne lève)
        si la génération/validation/exécution échoue à n'importe quelle
        étape, pour permettre à l'appelant de retomber sur le RAG vectoriel
        sans jamais casser la conversation.
        """
        if not self.should_attempt_sql(question):
            return None
        try:
            return self.sql_engine.answer(question, self.llm, file_hint=file_hint)
        except (SQLGenerationError, SQLSafetyError) as e:
            logger.info("Text-to-SQL path declined for %r: %s", question[:80], e)
            return None
        except Exception as e:
            logger.warning("Text-to-SQL path failed for %r: %s", question[:80], e)
            return None

    def answer(self, question: str) -> tuple[str, list[dict[str, Any]]]:
        """
        Répond à une question en passant par le pipeline text-to-SQL si elle
        est structurée/agrégée et qu'une base est disponible (résultat exact,
        sans hallucination possible sur les chiffres) ; sinon par le
        pipeline RAG complet (retrieval + génération).

        Returns:
            (reponse_generee, contextes_recuperes)
        """
        sql_result = self.try_sql_answer(question)
        if sql_result is not None:
            contexts = [
                {
                    "content": f"SQL executed on {sql_result.db_file}:\n{sql_result.sql}",
                    "score": 1.0,
                    "source": sql_result.db_file,
                    "page": "SQL",
                }
            ]
            return sql_result.answer_html, contexts

        # 1. Filtre par fichier si la question nomme explicitement un document
        #    (ex. « cherche dans CV_BENMOUH_Mohamed.pdf ») → recherche ciblée.
        where = _extract_document_filter(question, self.vector_store)

        # 2. Retrieval
        results = self.retriever.retrieve(
            question,
            top_k=self.top_k,
            min_score=self.min_score,
            where=where,
        )

        # 2. Formatage du contexte
        formatted_context = self.retriever.format_context(results)

        # 3. Génération de la réponse (si Ollama disponible)
        if self.llm:
            response = self.llm.generate(question, formatted_context)
        else:
            response = "[LLM non disponible - Ollama requis pour la génération]"

        # 4. Structuration des contextes pour Ragas
        contexts = [
            {
                "content": r.content,
                "score": r.score,
                "source": r.metadata.get("source_file", "inconnu"),
                "page": r.metadata.get("page_number", -1),
            }
            for r in results
        ]

        return response, contexts
