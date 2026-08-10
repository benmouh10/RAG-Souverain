"""
Stockage vectoriel local ChromaDB avec embeddings sentence-transformers.

Modèle par défaut : paraphrase-multilingual-MiniLM-L12-v2 (optimisé multilingue / FR).
"""

from __future__ import annotations

import json  # Ajoute pour serialiser les metadonnees complexes (ex: lignes SQL)
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_DIR = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _safe_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure metadata dict has no None/empty values for critical fields."""
    safe = dict(meta) if meta else {}
    sf = safe.get("source_file")
    safe["source_file"] = sf if sf and str(sf).strip() and str(sf).strip().lower() not in ("none",) else "unknown"
    pg = safe.get("page_number")
    safe["page_number"] = pg if pg is not None and str(pg).strip() and str(pg).strip().lower() not in ("none",) else 0
    ver = safe.get("version")
    safe["version"] = ver if ver is not None and str(ver).strip() and str(ver).strip().lower() not in ("none",) else 1
    return safe


@dataclass(frozen=True)
class SearchResult:
    """Résultat de recherche par similarité cosinus."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]


def _resolve_local_sentence_transformers_path(model_name: str, cache_root: str) -> str:
    if os.path.isdir(model_name):
        return model_name

    # Hugging Face local repository layout: ~/.cache/huggingface/hub/models--<repo-id>/snapshots/<hash>
    repo_folder = os.path.join(cache_root, "hub", "models--" + model_name.replace("/", "--"))
    if not os.path.isdir(repo_folder):
        return model_name

    snapshots_folder = os.path.join(repo_folder, "snapshots")
    if os.path.isdir(snapshots_folder):
        snapshots = [d for d in os.listdir(snapshots_folder) if os.path.isdir(os.path.join(snapshots_folder, d))]
        if snapshots:
            snapshots.sort(key=lambda d: os.path.getmtime(os.path.join(snapshots_folder, d)), reverse=True)
            return os.path.join(snapshots_folder, snapshots[0])

    return repo_folder


class LocalEmbeddingFunction:
    """Fonction d'embedding ChromaDB basée sur sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        logger.info("Chargement du modèle d'embeddings : %s", model_name)
        # Souveraineté : on résout d'abord un chemin local de modèle (huggingface
        # local snapshot, HF_HOME, ou dossier de snapshots), puis on force le mode
        # OFFLINE pour qu'aucune requête réseau ne soit émise au chargement.
        cache_root = (
            os.getenv("HF_HOME")
            or os.getenv("SENTENCE_TRANSFORMERS_HOME")
            or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        )
        model_path = _resolve_local_sentence_transformers_path(model_name, cache_root)

        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_root)
        os.environ.setdefault("HF_HOME", cache_root)
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_root)

        self._model = SentenceTransformer(
            model_path,
            cache_folder=cache_root,
            local_files_only=True,
            model_kwargs={"cache_dir": cache_root},
            config_kwargs={"cache_dir": cache_root},
            processor_kwargs={"cache_dir": cache_root},
        )
        self._model_name = model_name

    def name(self) -> str:
        return self._model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            input,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()


class ChromaVectorStore:
    """Interface ChromaDB persistante pour l'indexation et la recherche RAG."""

    def __init__(
        self,
        *,
        persist_directory: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_function: Any | None = None,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or LocalEmbeddingFunction(embedding_model)

        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection: Collection | None = None

    def initialize_collection(self, *, reset: bool = False) -> Collection:
        """Crée ou récupère la collection ChromaDB (similarité cosinus)."""
        if reset:
            try:
                self._client.delete_collection(name=self.collection_name)
                logger.info("Collection '%s' supprimée.", self.collection_name)
            except Exception:
                logger.debug("Aucune collection existante à supprimer.")

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Collection '%s' prête (%d documents indexés).",
            self.collection_name,
            self._collection.count(),
        )
        return self._collection

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            return self.initialize_collection()
        return self._collection

    def add_chunks(self, chunks: list[TextChunk], *, batch_size: int = 64) -> int:
        """Vectorise et injecte les chunks avec leurs metadonnees assainies."""
        if not chunks:
            return 0

        added = 0
        now_str = datetime.now(timezone.utc).isoformat()

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            ids = [str(uuid.uuid4()) for _ in batch]
            documents = [chunk.content for chunk in batch]
            
            # --- Zone corrigee : Assainissement des metadonnees ---
            metadatas = []
            for chunk in batch:
                raw_metadata = chunk.to_metadata()
                # Add ingested_at timestamp
                raw_metadata["ingested_at"] = now_str
                sanitized_metadata = {}
                
                for key, value in raw_metadata.items():
                    if isinstance(value, (dict, list)):
                        sanitized_metadata[key] = json.dumps(value, ensure_ascii=False)
                    elif value is None:
                        sanitized_metadata[key] = ""
                    else:
                        sanitized_metadata[key] = value
                        
                metadatas.append(sanitized_metadata)
            # ------------------------------------------------------

            embeddings = self.embedding_function(documents)

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            added += len(batch)
            logger.debug("Batch indexe : %d chunks.", len(batch))

        logger.info("%d chunk(s) indexe(s) dans '%s'.", added, self.collection_name)
        return added

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Recherche par similarité cosinus sur les embeddings normalisés."""
        query_embedding = self.embedding_function.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def search_latest(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_version: int | None = None,
    ) -> list[SearchResult]:
        """Recherche uniquement les chunks de la **dernière version indexée**.

        Correction C4 : la sémantique historique ``{"version": {"$lte": max_version}}``
        était incorrecte — elle retrouvait *toutes* les versions ≤ seuil, mélangeant
        les chunks obsolètes et courants d'un même fichier. La sémantique attendue
        est une **égalité** sur la version la plus récente.

        - Si ``max_version`` est fourni, filtre sur ``version == max_version``.
        - Sinon, détermine la version maximale présente dans la collection
          (via ``get_latest_version``) et filtre sur celle-ci.

        Note : les chunks sans métadonnée ``version`` (indexés sans versionnage,
        ex. ingests SQL ou uploads directs) sont ignorés ici ; utiliser
        ``search`` pour une recherche sans contrainte de version.
        """
        target_version = max_version
        if target_version is None:
            target_version = self.get_latest_version_all()
        if target_version is None:
            logger.info("search_latest : aucune version indexée — aucun résultat.")
            return []
        where_filter = {"version": target_version}
        return self.search(query, top_k=top_k, where=where_filter)

    def get_latest_version_all(self) -> int | None:
        """Version maximale présente dans la collection (None si vide)."""
        all_data = self.collection.get(include=["metadatas"])
        versions = [m.get("version", 0) for m in all_data["metadatas"]]
        return max(versions) if versions else None

    def get_latest_version(self, filename: str) -> int:
        """Retourne le numéro de version le plus élevé pour un fichier donné."""
        all_data = self.collection.get(include=["metadatas"])
        max_ver = 0
        for meta in all_data["metadatas"]:
            if meta.get("source_file") == filename:
                v = meta.get("version", 1)
                if v > max_ver:
                    max_ver = v
        return max_ver

    def count_by_version(self, filename: str, version: int) -> int:
        """Compte les chunks pour un fichier et une version donnés.

        `collection.get(where=...)` n'accepte pas un dict multi-clés (ce serait
        interprété comme un opérateur) : on exprime la conjonction via ``$and``.
        """
        all_data = self.collection.get(
            include=["metadatas"],
            where={"$and": [{"source_file": filename}, {"version": version}]},
        )
        return len(all_data["ids"])

    @staticmethod
    def _format_results(results: dict[str, Any]) -> list[SearchResult]:
        formatted: list[SearchResult] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            # ChromaDB renvoie des distances ; comme l'espace HNSW est la
            # similarité cosinus, on convertit distance -> score (1 - d), puis
            # on borne dans [0, 1] pour les éventuels débordements d'arrondi.
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            formatted.append(
                SearchResult(
                    id=doc_id,
                    content=content or "",
                    score=score,
                    metadata=_safe_meta(metadata),
                )
            )

        return formatted

    def count(self) -> int:
        return self.collection.count()

    def get_source_files(self) -> set[str]:
        """Ensemble des noms de fichiers distincts indexés dans la collection."""
        metadatas = self.collection.get(include=["metadatas"]).get("metadatas", [])
        return {_safe_meta(m).get("source_file", "unknown") for m in metadatas}

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Return all stored documents with their IDs, texts, and metadata.
        Ensures metadata always contains at least source_file and page_number."""
        all_data = self.collection.get(include=["documents", "metadatas"])
        docs = []
        for doc_id, text, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            docs.append({
                "id": doc_id,
                "text": text or "",
                "metadata": _safe_meta(meta),
            })
        return docs

    def delete_by_source(self, source_filename: str) -> int:
        """Delete all chunks from a specific source file. Returns count deleted."""
        try:
            all_data = self.collection.get(include=["metadatas"])
            ids_to_delete = [
                doc_id for doc_id, meta in zip(all_data["ids"], all_data["metadatas"])
                if meta.get("source_file") == source_filename
            ]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info("Deleted %d chunks from source '%s'", len(ids_to_delete), source_filename)
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error("Failed to delete by source '%s': %s", source_filename, e)
            return 0

    def list_sources(self) -> list[dict[str, Any]]:
        """Return distinct source filenames with their ingestion dates and chunk counts."""
        all_data = self.collection.get(include=["metadatas"])
        sources: dict[str, dict[str, Any]] = {}
        for meta in all_data["metadatas"]:
            src = meta.get("source_file", "unknown")
            ingested_at = meta.get("ingested_at", "")
            version = meta.get("version", 1)
            key = f"{src}_v{version}"
            if key not in sources:
                sources[key] = {
                    "source_file": src,
                    "version": version,
                    "chunk_count": 0,
                    "ingested_at": ingested_at,
                }
            sources[key]["chunk_count"] += 1
        return sorted(sources.values(), key=lambda d: (d["source_file"], -d["version"]))

    def clear(self) -> int:
        """Supprime tous les documents de la collection ChromaDB et retourne le nouveau comptage."""
        try:
            self.collection.delete(where={})
        except Exception:
            try:
                self._client.delete_collection(name=self.collection_name)
                self.initialize_collection()
            except Exception:
                logger.exception("Échec lors de la purge de la collection ChromaDB.")
                raise
        return self.collection.count()