"""Tests d'intégration du pipeline d'embeddings avec le modèle réel.

Exclus du CI rapide (pytest.ini : ``-m "not integration"``). À exécuter dans
l'image Docker (modèle pré-chargé) ou localement avec le cache HuggingFace :

    pytest -m integration tests/test_integration_embeddings.py -v
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2


def test_local_embedding_function_batch_shape():
    """Le modèle réel vectorise un lot de textes en 384 dimensions."""
    from app.indexing.vector_store import LocalEmbeddingFunction

    ef = LocalEmbeddingFunction()
    vectors = ef(["bonjour", "le rag souverain fonctionne en circuit ferme"])

    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_local_embedding_function_query_shape():
    """embed_query produit une requête vectorielle de même dimension."""
    from app.indexing.vector_store import LocalEmbeddingFunction

    ef = LocalEmbeddingFunction()
    q = ef.embed_query("question de test en français")

    assert len(q) == EMBEDDING_DIM


def test_local_embedding_function_normalized():
    """Les embeddings sont normalisés (norme L2 ≈ 1) pour la similarité cosinus."""
    import math

    from app.indexing.vector_store import LocalEmbeddingFunction

    ef = LocalEmbeddingFunction()
    vector = ef(["contenu de test"])[0]

    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) < 1e-3
