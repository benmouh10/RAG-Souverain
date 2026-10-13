"""
Module de recherche (Retrieval) pour le systeme RAG.

Combines vector search (ChromaDB cosine) + BM25 keyword search
with Reciprocal Rank Fusion, then CrossEncoder reranking.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any

from app.indexing.vector_store import ChromaVectorStore, SearchResult, _safe_meta

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CrossEncoder reranker — loaded once at module level
# ---------------------------------------------------------------------------
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_cross_encoder = None


def _get_cross_encoder():
    """Lazy-load CrossEncoder (downloaded once, cached in HF_HOME).

    Chargement **hors-ligne** : on résout d'abord le snapshot local via
    ``_resolve_local_sentence_transformers_path`` (même mécanisme que le modèle
    d'embeddings), sinon ``CrossEncoder(repo_id)`` tenterait un
    ``snapshot_download`` réseau qui échoue en mode OFFLINE.
    """
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder

            from app.indexing.vector_store import _resolve_local_sentence_transformers_path

            cache_root = os.getenv("HF_HOME", os.getenv("SENTENCE_TRANSFORMERS_HOME", ""))
            model_path = (
                _resolve_local_sentence_transformers_path(_RERANKER_MODEL, cache_root)
                if cache_root
                else _RERANKER_MODEL
            )
            logger.info("Loading CrossEncoder reranker: %s (local path: %s)", _RERANKER_MODEL, model_path)
            _cross_encoder = CrossEncoder(model_path, max_length=512)
            logger.info("CrossEncoder reranker loaded successfully.")
        except Exception as e:
            logger.warning(
                "Failed to load CrossEncoder reranker (%s): %s. "
                "Falling back to cosine similarity ranking.",
                _RERANKER_MODEL, e,
            )
            _cross_encoder = False  # sentinel: tried and failed
    return _cross_encoder if _cross_encoder is not False else None


# ---------------------------------------------------------------------------
# BM25 index — lazy-loaded, rebuilt when collection changes
# ---------------------------------------------------------------------------
_bm25_index = None
_bm25_docs: list[dict[str, Any]] = []
_bm25_collection_count = 0
USE_HYBRID = os.getenv("USE_HYBRID_SEARCH", "true").lower() in ("true", "1", "yes")
USE_HYDE = os.getenv("USE_HYDE", "false").lower() in ("true", "1", "yes")
# Poids du reranker CrossEncoder dans la fusion finale (0.0 → RRF pur,
# 1.0 → CE pur). Le CE ms-marco est entraîné sur de l'anglais : sur du
# contenu français il peut hisser des chunks non pertinents en tête. On
# garde le RRF comme signal dominant et on n'utilise le CE qu'en réordonneur
# léger (valeur par défaut raisonnable pour des documents multilingues).
RERANK_CE_WEIGHT = float(os.getenv("RERANK_CE_WEIGHT", "0.35"))
_BM25_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bm25_cache.pkl"


_ACCENT_LIGATURES = str.maketrans(
    {"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae"}
)

_FRENCH_ACCENTS = str.maketrans(
    "àâäáãåéèêëìíîïòóôöõùúûüçñ",
    "aaaaaaeeeeiiiiooooouuuucn",
)


def _strip_accents(text: str) -> str:
    """Retrait des accents pour un rapprochement BM25 insensible aux variantes
    typographiques françaises (« génération » == « generation »)."""
    import unicodedata

    normalized = unicodedata.normalize("NFKD", text).translate(_ACCENT_LIGATURES)
    cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
    return cleaned.translate(_FRENCH_ACCENTS)


def _tokenize(text: str) -> list[str]:
    """Tokenisation BM25 : minuscules + mots, avec variantes sans accents
    pour ne pas rater les requêtes à orthographe francisée."""
    tokens = re.findall(r"\w+", text.lower())
    out: list[str] = []
    for token in tokens:
        out.append(token)
        stripped = _strip_accents(token)
        if stripped != token:
            out.append(stripped)
    return out


def _save_bm25_cache(index: Any, docs: list[dict[str, Any]], count: int) -> None:
    try:
        _BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_BM25_CACHE_PATH, "wb") as f:
            pickle.dump({"index": index, "docs": docs, "count": count}, f)
        logger.info("Saved BM25 index cache to %s (%d docs)", _BM25_CACHE_PATH, count)
    except Exception as e:
        logger.warning("Failed to save BM25 cache: %s", e)


def _load_bm25_cache(expected_count: int) -> bool:
    global _bm25_index, _bm25_docs, _bm25_collection_count
    if not _BM25_CACHE_PATH.exists():
        return False
    try:
        with open(_BM25_CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        if data.get("count") == expected_count and data.get("index") is not None:
            _bm25_index = data["index"]
            _bm25_docs = data.get("docs", [])
            _bm25_collection_count = data["count"]
            logger.info("Loaded BM25 index from cache (%d docs)", expected_count)
            return True
    except Exception as e:
        logger.warning("Failed to load BM25 cache: %s", e)
    return False


def _build_bm25_index(vector_store: ChromaVectorStore):
    """Build or rebuild the BM25 index from all stored documents."""
    global _bm25_index, _bm25_docs, _bm25_collection_count

    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.error(
            "rank-bm25 n'est PAS installé : la recherche hybride (vector + BM25) "
            "est désactivée, seule la recherche vectorielle est active. "
            "Installez-le avec `pip install rank-bm25` (présent dans requirements.txt)."
        )
        return

    all_docs = vector_store.get_all_documents()
    if not all_docs:
        logger.info("No documents in collection. BM25 index empty.")
        _bm25_index = None
        _bm25_docs = []
        _bm25_collection_count = 0
        return

    corpus_tokens = [_tokenize(doc["text"]) for doc in all_docs]
    if not corpus_tokens:
        logger.info("Empty corpus. BM25 index empty.")
        _bm25_index = None
        _bm25_docs = []
        _bm25_collection_count = 0
        return

    _bm25_index = BM25Okapi(corpus_tokens)
    _bm25_docs = all_docs
    _bm25_collection_count = len(all_docs)
    logger.info("BM25 index built: %d documents.", len(all_docs))
    _save_bm25_cache(_bm25_index, _bm25_docs, _bm25_collection_count)


def _ensure_bm25_index(vector_store: ChromaVectorStore):
    """Rebuild BM25 index if collection has changed, loading disk cache if available."""
    global _bm25_index, _bm25_collection_count
    current_count = vector_store.count()
    if _bm25_index is not None and current_count == _bm25_collection_count:
        return
    if _load_bm25_cache(current_count):
        return
    _build_bm25_index(vector_store)


# ---------------------------------------------------------------------------
# Expansion d'acronymes — glossaire dérivé du corpus
# ---------------------------------------------------------------------------
# Certaines questions définitionnelles (« Qu'est-ce que le système RAG ? »)
# échouent à retrouver l'entrée du glossaire : le terme seul ne suffit pas à
# rapprocher l'embedding de la définition. On construit — à partir du corpus
# indexé — une table terme → expansion (ex. « RAG » → « Retrieval-Augmented
# Generation ») et on l'injecte dans la requête de recherche pour les seules
# questions définitionnelles. Aucun terme n'est codé en dur : la table est
# dérivée des documents au premier usage.
_acronym_map: dict[str, str] | None = None
_acronym_map_count = 0

# Une entrée de glossaire = une ligne « TERME » suivie d'une ligne contenant
# un tiret cadratin (« — »), l'expansion étant le texte avant le tiret.
_GLOSSARY_TERM_RE = re.compile(r"^[A-ZÀ-Ü][A-ZÀ-Ü0-9 \-]{1,14}$")

_DEFINITION_QUERY_RE = re.compile(
    r"(qu'?est-?ce que|c'est quoi|what is|que signifie|défini)", re.IGNORECASE
)


def _extract_acronym_map(docs: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for doc in docs:
        lines = (doc.get("text") or "").splitlines()
        for i, line in enumerate(lines[:-1]):
            term = line.strip()
            if not _GLOSSARY_TERM_RE.match(term):
                continue
            nxt = lines[i + 1].strip()
            if "—" not in nxt and "–" not in nxt:
                continue
            expansion = re.split(r"[—–]", nxt, maxsplit=1)[0].strip()
            if not expansion or expansion.lower() == term.lower():
                continue
            mapping.setdefault(term, expansion[:60])
    return mapping


def _ensure_acronym_map(vector_store: ChromaVectorStore) -> dict[str, str]:
    """(Re)construit la table terme → expansion quand la collection change."""
    global _acronym_map, _acronym_map_count
    current_count = vector_store.count()
    if _acronym_map is not None and current_count == _acronym_map_count:
        return _acronym_map
    _acronym_map = _extract_acronym_map(vector_store.get_all_documents())
    _acronym_map_count = current_count
    if _acronym_map:
        logger.info("Expansion d'acronymes prête (%d termes).", len(_acronym_map))
    return _acronym_map


def _expand_query_acronyms(
    query: str, vector_store: ChromaVectorStore
) -> str:
    """Enrichit la requête avec l'expansion des termes qu'elle contient —
    uniquement pour les questions définitionnelles (ex. « Qu'est-ce que le
    système RAG ? » → ajoute « Retrieval-Augmented Generation »)."""
    if not _DEFINITION_QUERY_RE.search(query):
        return query
    mapping = _ensure_acronym_map(vector_store)
    if not mapping:
        return query
    additions: list[str] = []
    for term, expansion in mapping.items():
        if re.search(rf"\b{re.escape(term)}\b", query):
            additions.append(expansion)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given rank (1-indexed)."""
    return 1.0 / (k + rank)


class Retriever:
    """
    Recupere le contexte pertinent depuis le Vector Store en fonction d'une requete.
    Combines vector search + BM25 keyword search + CrossEncoder reranking + optional HyDE.
    """

    def __init__(self, vector_store: ChromaVectorStore, llm: Any | None = None) -> None:
        self.vector_store = vector_store
        self._llm = llm

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        min_score: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Recherche les chunks les plus pertinents pour une requete donnee.

        Steps:
        1. (Optional) HyDE: generate hypothetical answer, use as search text
        2. Vector search from ChromaDB (top_k * 2 + 2 candidates)
        3. BM25 keyword search (if hybrid enabled)
        4. Merge with Reciprocal Rank Fusion
        5. Filter by min_score
        6. Rerank with CrossEncoder if available
        7. Return top_k results
        """
        logger.info("Recherche pour: '%s' (top_k=%d, min_score=%.2f)", query, top_k, min_score)

        fetch_count = top_k * 2 + 2

        # Step 1: HyDE query expansion
        if USE_HYDE and self._llm is not None:
            try:
                search_text = self._llm.generate_hypothetical_answer(query)
                logger.info("HyDE: expanded query -> '%s'", search_text[:80])
            except Exception as e:
                logger.warning("HyDE expansion failed: %s. Using raw query.", e)
                search_text = query
        else:
            search_text = query

        # Step 1bis: expansion d'acronymes du corpus (glossaire) pour les
        # questions définitionnelles — « RAG » → « Retrieval-Augmented
        # Generation ». Améliore nettement le rappel des définitions exactes.
        expanded = _expand_query_acronyms(search_text, self.vector_store)
        if expanded != search_text:
            logger.info("Query expansion: '%s'", expanded[:160])
            search_text = expanded

        # Step 2: Vector search from ChromaDB
        vector_results = self.vector_store.search(search_text, top_k=fetch_count, where=where)

        if USE_HYBRID:
            # Step 2: BM25 keyword search
            _ensure_bm25_index(self.vector_store)
            bm25_results = self._bm25_search(search_text, top_k=fetch_count)

            # Un filtre ``where`` (fichier nommé dans la question) ne s'applique
            # qu'à la recherche vectorielle côté ChromaDB : on restreint aussi les
            # résultats BM25 pour ne pas réinjecter d'autres fichiers via la fusion.
            if where is not None:
                wanted = where.get("source_file")
                bm25_results = [
                    r for r in bm25_results
                    if r.metadata.get("source_file") == wanted
                ]

            # Step 3: Merge with RRF
            merged = self._rrf_merge(vector_results, bm25_results, top_k=fetch_count)
        else:
            merged = vector_results

        # Step 4: Filter by min_score
        filtered = [r for r in merged if r.score >= min_score]

        if not filtered:
            logger.warning(
                "Aucun chunk passe le seuil min_score=%.2f.", min_score,
            )
            return []

        # Step 4bis: déduplication sur le contenu — le chunking de certains
        # fichiers (PDF court, README) produit N chunks identiques qui
        # monopolisent les top_k et polluent le contexte du LLM.
        seen_content: set[str] = set()
        deduped: list[SearchResult] = []
        for r in filtered:
            key = r.content.strip().lower()
            if key and key not in seen_content:
                seen_content.add(key)
                deduped.append(r)
        filtered = deduped
        logger.info("Déduplication retrieval: %d chunks uniques.", len(filtered))

        # Step 5: Rerank with CrossEncoder
        cross_encoder = _get_cross_encoder()
        if cross_encoder is not None and len(filtered) > 1:
            # Check if reranking is enabled via UI toggle (session_state)
            reranker_enabled = True
            try:
                import streamlit as st
                reranker_enabled = st.session_state.get("toggle_reranker", True)
            except Exception:
                pass
            if reranker_enabled:
                try:
                    pairs = [(query, r.content) for r in filtered]
                    ce_scores = cross_encoder.predict(pairs)

                    # Normalisation min-max des scores CE dans [0, 1].
                    ce_min, ce_max = min(ce_scores), max(ce_scores)
                    ce_norm = [
                        (s - ce_min) / (ce_max - ce_min) if ce_max > ce_min else 0.5
                        for s in ce_scores
                    ]

                    # Fusion douce score_final = (1-w)*RRF + w*CE. Le score RRF
                    # (post-min-max) est déjà dans [0,1] et reflète fidèlement
                    # la pertinence hybride (vecteur + BM25) ; le CE anglais est
                    # volontairement pondéré faiblement pour ne pas dégrader le
                    # français (ex. README bruité hissé en tête).
                    blended = [
                        (1.0 - RERANK_CE_WEIGHT) * r.score + RERANK_CE_WEIGHT * c
                        for r, c in zip(filtered, ce_norm)
                    ]
                    reranked = sorted(
                        zip(blended, filtered), key=lambda x: x[0], reverse=True
                    )
                    final = [r for _, r in reranked[:top_k]]
                    logger.info(
                        "CrossEncoder reranking (w=%.2f): %d -> %d results.",
                        RERANK_CE_WEIGHT, len(filtered), len(final),
                    )
                    return final
                except Exception as e:
                    logger.warning("CrossEncoder reranking failed: %s. Using merged ranking.", e)

        return filtered[:top_k]

    def _bm25_search(self, query: str, top_k: int) -> list[SearchResult]:
        """Run BM25 keyword search over stored documents.
        After BM25 ranks indices, fetches full metadata from ChromaDB."""
        if _bm25_index is None or not _bm25_docs:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        try:
            scores = _bm25_index.get_scores(query_tokens)
        except Exception as e:
            logger.warning("BM25 scoring failed: %s", e)
            return []

        # Get top-k indices sorted by BM25 score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Collect IDs of documents that passed BM25 ranking
        bm25_ids = []
        bm25_scores = {}
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            doc = _bm25_docs[idx]
            bm25_ids.append(doc["id"])
            bm25_scores[doc["id"]] = scores[idx]

        if not bm25_ids:
            return []

        # Fetch full metadata from ChromaDB to avoid None fields
        try:
            fetched = self.vector_store.collection.get(
                ids=bm25_ids,
                include=["documents", "metadatas"],
            )
            id_to_meta: dict[str, dict[str, Any]] = {}
            id_to_text: dict[str, str] = {}
            for fid, ftext, fmeta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"]):
                id_to_meta[fid] = _safe_meta(fmeta)
                id_to_text[fid] = ftext or ""
        except Exception as e:
            logger.warning("ChromaDB metadata fetch failed for BM25 results: %s", e)
            # Fallback to cached metadata from _bm25_docs
            id_to_meta = {doc["id"]: doc["metadata"] for doc in _bm25_docs if doc["id"] in bm25_ids}
            id_to_text = {doc["id"]: doc["text"] for doc in _bm25_docs if doc["id"] in bm25_ids}

        max_possible = max(bm25_scores.values()) if bm25_scores else 1.0
        results = []
        for doc_id in bm25_ids:
            normalized_score = float(bm25_scores[doc_id]) / float(max_possible)
            meta = id_to_meta.get(doc_id, {"source_file": "unknown", "page_number": 0})
            meta = _safe_meta(meta)
            results.append(SearchResult(
                id=doc_id,
                content=id_to_text.get(doc_id, ""),
                score=normalized_score,
                metadata=meta,
            ))
        return results

    @staticmethod
    def _rrf_merge(
        vector_results: list[SearchResult],
        bm25_results: list[SearchResult],
        *,
        top_k: int,
        k: int = 60,
    ) -> list[SearchResult]:
        """Merge vector and BM25 results using Reciprocal Rank Fusion."""
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, SearchResult] = {}

        # Vector results contribute RRF scores
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.id
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + _rrf_score(rank, k)
            doc_map[doc_id] = result

        # BM25 results contribute RRF scores
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result.id
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + _rrf_score(rank, k)
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # Sort by combined RRF score descending
        sorted_ids = sorted(doc_scores.keys(), key=lambda did: doc_scores[did], reverse=True)[:top_k]

        # Normalize scores to [0, 1] so min_score filtering works correctly
        if sorted_ids:
            max_rrf = max(doc_scores[did] for did in sorted_ids)
            min_rrf = min(doc_scores[did] for did in sorted_ids)
            span = max_rrf - min_rrf if max_rrf != min_rrf else 1.0
        else:
            max_rrf = min_rrf = span = 1.0

        merged = []
        for doc_id in sorted_ids:
            result = doc_map[doc_id]
            rrf = doc_scores[doc_id]
            normalized = (rrf - min_rrf) / span
            merged.append(SearchResult(
                id=result.id,
                content=result.content,
                score=normalized,
                metadata=_safe_meta(result.metadata),
            ))
        return merged

    def format_context(self, results: list[SearchResult]) -> str:
        """Formate les resultats recuperes en un bloc de texte clair pour le LLM."""
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, start=1):
            source = result.metadata.get("source_file", "Source inconnue")
            page = result.metadata.get("page_number")
            sec = result.metadata.get("section_title")
            
            page_info = f" - page {page}" if page and page != -1 else ""
            sec_info = f" | Section: {sec}" if sec else ""
            header = f"[Extrait {i} - {source}{page_info}{sec_info}]"
            
            context_parts.append(f"{header}\n{result.content.strip()}")

        return "\n\n".join(context_parts)
