"""
Pipeline d'évaluation automatisé RAGAS — Sprint 7 (Évaluation)
==============================================================

Ce module implémente un pipeline d'évaluation complet pour mesurer
scientifiquement la qualité du RAG Souverain à l'aide du framework 'ragas'.

Principes :
- 100% local : utilise Ollama comme LLM de notation (Evaluator/Judge)
- Aucun appel à l'API OpenAI : client OpenAI-compatible pointant sur localhost
- Mesure 4 métriques fondamentales : Faithfulness, Answer Relevancy,
  Context Precision, Context Recall
- Export des résultats en CSV pour analyse

Utilisation :
    python -m app.evaluation.eval_pipeline

Prérequis :
    - Ollama doit être en cours d'exécution (ollama serve)
    - Le modèle mistral (ou autre) doit être téléchargé (ollama pull mistral)
    - Les documents doivent être indexés dans ChromaDB
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeoutError
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Configuration du logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_pipeline")

# ---------------------------------------------------------------------------
# Imports RAGAS (avec fallback OpenAI local)
# ---------------------------------------------------------------------------
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Imports de notre code RAG
# ---------------------------------------------------------------------------
# Le moteur RAG (RAGEngine) vit dans la couche cœur (app.core.rag_engine),
# voir remaniement C2. Ce module d'évaluation n'en est plus qu'un consommateur.
from app.core.rag_engine import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    GENERATOR_MODEL,
    OLLAMA_BASE_URL,
    RAGEngine,
)

# ---------------------------------------------------------------------------
# Configuration (variables d'environnement pour Docker)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

EVALUATOR_MODEL = "mistral"  # Modèle utilisé comme juge RAGAS
REPORT_PATH = Path(os.getenv("EVALUATION_REPORT_PATH", str(PROJECT_ROOT / "data" / "evaluation_report.csv")))
QUERY_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_QUERY_TIMEOUT", "300"))  # per-question budget
RAGAS_TIMEOUT_SECONDS = int(os.getenv("EVALUATION_RAGAS_TIMEOUT", "300"))  # budget des métriques Ragas
# Timeout interne à Ragas (run_config) : nettement au-dessus du budget pipeline
# pour ne pas masquer les erreurs réelles — Ollama est lent (génération CPU).
RAGAS_RUN_CONFIG_TIMEOUT = int(os.getenv("EVALUATION_RAGAS_RUN_TIMEOUT", "600"))

# M4 — Observabilité : la boucle d'évaluation alimente la même table query_logs
# que le chat, afin que les analyses de performance couvrent l'ensemble du trafic.
from app.observability import QueryLogger

_QUERY_LOGGER = QueryLogger()


def _run_with_timeout(fn: Callable[[], Any], timeout: float, default: Any = None) -> Any:
    """Run ``fn`` in a worker thread and return its result or ``default`` on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FutTimeoutError:
            return default
        except Exception:
            raise


# ===========================================================================
# 1. GOLD DATASET — 30 questions/réponses de référence
from app.evaluation.gold_dataset import GOLD_DATASET


# ===========================================================================
# 2. CLIENT RAG COMPLET
# ===========================================================================
# La classe RAGEngine a été déplacée dans app.core.rag_engine (remaniement C2) :
#   from app.core.rag_engine import RAGEngine
# Ce module ne conserve que l'évaluation RAGAS et ses helpers.


# ===========================================================================
# 3. ÉVALUATEUR RAGAS LOCAL (optionnel, nécessite Ollama)
# ===========================================================================

def setup_ragas_evaluator():
    """
    Configure Ragas pour utiliser Ollama comme LLM de notation (Evaluator/Judge).
    Retourne les métriques ou None si Ollama n'est pas disponible.

    Mode retrieval-only : poser ``EVALUATION_RETRIEVAL_ONLY=1`` pour collecter
    uniquement les réponses + contextes (rapide, sans dépendance Ragas) — utile
    quand ragas/langchain sont incompatibles ou pour un test démonstratif.
    """
    if os.getenv("EVALUATION_RETRIEVAL_ONLY", "0") == "1":
        logger.info("Mode retrieval-only activé (EVALUATION_RETRIEVAL_ONLY=1) : "
                    "aucune métrique Ragas ne sera calculée.")
        return None

    # Vérifier qu'Ollama est accessible
    import requests
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=15)
        if resp.status_code != 200:
            logger.warning("Ollama non disponible (status %d)", resp.status_code)
            return None
    except Exception as e:
        logger.warning("Ollama non disponible: %s", e)
        return None

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
    except ImportError as exc:
        # Ex. ragas 0.4.x avec langchain v1 : l'import échoue (module
        # langchain_community.chat_models.vertexai supprimé). L'évaluation
        # dégrade proprement en mode retrieval-only plutôt que de planter.
        logger.warning(
            "RAGAS indisponible (dépendances incompatibles) : %s. "
            "Seules les données de retrieval seront collectées.",
            exc,
        )
        return None

    # Client OpenAI-compatible pointant sur l'URL locale d'Ollama
    langchain_llm = ChatOpenAI(
        model=EVALUATOR_MODEL,
        api_key="ollama",
        base_url=f"{OLLAMA_BASE_URL}/v1",
        temperature=0.0,
    )

    # Création du LLM de notation Ragas via LangChain
    evaluator_llm = LangchainLLMWrapper(langchain_llm)

    local_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"),
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    # Attribution du LLM à chaque métrique
    faithfulness.llm = evaluator_llm
    answer_relevancy.llm = evaluator_llm
    answer_relevancy.embeddings = evaluator_embeddings
    context_precision.llm = evaluator_llm
    context_recall.llm = evaluator_llm

    logger.info("RAGAS configuré avec le modèle '%s' sur %s/v1",
                EVALUATOR_MODEL, OLLAMA_BASE_URL)

    return [faithfulness, answer_relevancy, context_precision, context_recall]


# ===========================================================================
# 4. BOUCLE D'ÉVALUATION
# ===========================================================================

@dataclass
class EvalResult:
    """Résultat d'évaluation pour un cas de test."""
    question: str
    reference: str
    response: str
    contexts: list[dict[str, Any]]
    faithfulness: float = -1.0
    answer_relevancy: float = -1.0
    context_precision: float = -1.0
    context_recall: float = -1.0
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def metrics_available(self) -> bool:
        return self.faithfulness >= 0


def run_evaluation(
    progress_callback: Callable[[int, int], None] | None = None,
    timeout_seconds: int | None = None,
) -> list[EvalResult]:
    """
    Exécute la boucle d'évaluation complète :
    1. Pour chaque question du Gold Dataset
    2. Interroge le RAG (avec timeout par question)
    3. Calcule les métriques Ragas (si Ollama disponible, avec timeout)
    4. Agrège les résultats

    Args:
        progress_callback: called as ``progress_callback(done, total)``.
        timeout_seconds: strict budget per question (defaults to QUERY_TIMEOUT_SECONDS).
    """
    logger.info("=" * 70)
    logger.info("DÉMARRAGE DE L'ÉVALUATION RAGAS — Sprint 7")
    logger.info("Dataset: %d questions de test", len(GOLD_DATASET))
    logger.info("Modèle générateur: %s", GENERATOR_MODEL)
    logger.info("Timeout par question: %ds", timeout_seconds or QUERY_TIMEOUT_SECONDS)
    logger.info("=" * 70)

    # Initialisation du moteur RAG
    rag = RAGEngine()

    # Configuration de Ragas
    metrics = setup_ragas_evaluator()
    if metrics:
        logger.info("✓ Évaluation RAGAS complète avec le modèle '%s'", EVALUATOR_MODEL)
    else:
        logger.warning("⚠ RAGAS non disponible (Ollama requis). "
                       "Seules les données de retrieval seront collectées.")

    # Résultats
    results: list[EvalResult] = []
    budget = timeout_seconds or QUERY_TIMEOUT_SECONDS

    # Boucle sur chaque question
    for i, test_case in enumerate(GOLD_DATASET, start=1):
        question = test_case["user_input"]
        reference = test_case["reference"]

        logger.info("[%d/%d] %s", i, len(GOLD_DATASET),
                    question[:80] + ("..." if len(question) > 80 else ""))

        result = EvalResult(
            question=question,
            reference=reference,
            response="",
            contexts=[],
        )

        try:
            start_time = time.time()

            # Étape 1 : Obtenir la réponse et les contextes du RAG
            def _answer() -> tuple[str, list[dict[str, Any]]]:
                return rag.answer(question)

            answered = _run_with_timeout(_answer, timeout=budget)
            if answered is None:
                raise TimeoutError(f"Timeout dépassé ({budget}s) pour la question.")

            response, contexts = answered
            result.response = response
            result.contexts = contexts
            result.latency_ms = (time.time() - start_time) * 1000

            # M4 — Log la requête dans query_logs (même observabilité que le chat)
            top1_score = contexts[0]["score"] if contexts else 0.0
            mean_score = (
                sum(c["score"] for c in contexts) / len(contexts) if contexts else 0.0
            )
            _QUERY_LOGGER.log(
                question=question,
                top1_score=top1_score,
                mean_score=mean_score,
                num_chunks=len(contexts),
                llm_latency_ms=result.latency_ms,
                total_latency_ms=result.latency_ms,
                model_used=GENERATOR_MODEL,
                source_files=[c["source"] for c in contexts],
            )

            logger.info("  → %d contextes, %.0f ms", len(contexts), result.latency_ms)

            # Étape 2 : Calcul des métriques Ragas (si disponible)
            if metrics and response:
                def _score() -> tuple[float, float, float, float]:
                    from datasets import Dataset
                    from ragas import evaluate
                    from ragas.run_config import RunConfig

                    eval_dataset = Dataset.from_dict({
                        "question": [question],
                        "answer": [response],
                        "contexts": [[c["content"] for c in contexts]],
                        "ground_truth": [reference],
                    })

                    scores = evaluate(
                        dataset=eval_dataset,
                        metrics=metrics,
                        run_config=RunConfig(timeout=RAGAS_RUN_CONFIG_TIMEOUT),
                    )

                    scores_df = scores.to_pandas()
                    first_row = scores_df.iloc[0]
                    return (
                        float(first_row.get("faithfulness", 0.0)),
                        float(first_row.get("answer_relevancy", 0.0)),
                        float(first_row.get("context_precision", 0.0)),
                        float(first_row.get("context_recall", 0.0)),
                    )

                scored = _run_with_timeout(_score, timeout=RAGAS_TIMEOUT_SECONDS)
                if scored is None:
                    logger.warning("  → Ragas timed out (>%ds) — skipping metrics.", RAGAS_TIMEOUT_SECONDS)
                    result.error = f"Ragas timeout dépassé ({RAGAS_TIMEOUT_SECONDS}s)"
                else:
                    (result.faithfulness, result.answer_relevancy,
                     result.context_precision, result.context_recall) = scored
                    logger.info("  → Faith=%.3f Rel=%.3f Prec=%.3f Recall=%.3f",
                                result.faithfulness, result.answer_relevancy,
                                result.context_precision, result.context_recall)

        except TimeoutError as e:
            logger.error("  → TIMEOUT: %s", str(e))
            result.error = str(e)
        except Exception as e:
            logger.error("  → ERREUR: %s", str(e))
            result.error = str(e)

        results.append(result)
        logger.info("✅ Question %d/%d treated. %d questions left.", i, len(GOLD_DATASET), len(GOLD_DATASET) - i)
        if progress_callback is not None:
            progress_callback(i, len(GOLD_DATASET))

        time.sleep(0.3)

    return results


# ===========================================================================
# 5. EXPORT ET RAPPORT
# ===========================================================================

def export_results(results: list[EvalResult], output_path: str | Path = REPORT_PATH):
    """Exporte les résultats détaillés au format CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "question", "reference", "response",
        "faithfulness", "answer_relevancy",
        "context_precision", "context_recall",
        "latency_ms", "error", "nb_contexts",
        "top1_score", "top1_source",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            top1 = r.contexts[0] if r.contexts else {}
            writer.writerow({
                "question": r.question,
                "reference": r.reference,
                "response": r.response,
                "faithfulness": f"{r.faithfulness:.4f}" if r.metrics_available else "N/A",
                "answer_relevancy": f"{r.answer_relevancy:.4f}" if r.metrics_available else "N/A",
                "context_precision": f"{r.context_precision:.4f}" if r.metrics_available else "N/A",
                "context_recall": f"{r.context_recall:.4f}" if r.metrics_available else "N/A",
                "latency_ms": f"{r.latency_ms:.1f}",
                "error": r.error or "",
                "nb_contexts": len(r.contexts),
                "top1_score": f"{top1.get('score', 0):.4f}" if top1 else "",
                "top1_source": top1.get("source", "") if top1 else "",
            })

    logger.info("Rapport exporté : %s (%d questions)", output_path, len(results))


def print_summary(results: list[EvalResult]):
    """Affiche un résumé statistique clair dans la console."""
    valid_results = [r for r in results if r.error is None]
    metrics_available = [r for r in valid_results if r.metrics_available]

    print()
    print("=" * 70)
    print("  RAPPORT D'ÉVALUATION RAGAS — RAG SOUVERAIN")
    print("=" * 70)
    print(f"  Date:                {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Modèle:              {GENERATOR_MODEL} (Ollama)")
    print(f"  Cas de test:         {len(results)}")
    print(f"  Réussis:             {len(valid_results)}")
    print(f"  Erreurs:             {len(results) - len(valid_results)}")
    print()

    if metrics_available:
        # Calcul des moyennes
        avg_faith = sum(r.faithfulness for r in metrics_available) / len(metrics_available)
        avg_rel = sum(r.answer_relevancy for r in metrics_available) / len(metrics_available)
        avg_prec = sum(r.context_precision for r in metrics_available) / len(metrics_available)
        avg_recall = sum(r.context_recall for r in metrics_available) / len(metrics_available)
        avg_latency = sum(r.latency_ms for r in valid_results) / len(valid_results)

        print("  " + "-" * 66)
        print(f"  {'Métrique':<25} {'Moyenne':<10} {'Cible':<10} {'Statut':<15}")
        print("  " + "-" * 66)
        print(f"  {'Faithfulness':<25} {avg_faith:<10.4f} {'≥ 0.80':<10} "
              f"{'✓' if avg_faith >= 0.80 else '✗':<15}")
        print(f"  {'Answer Relevancy':<25} {avg_rel:<10.4f} {'≥ 0.70':<10} "
              f"{'✓' if avg_rel >= 0.70 else '✗':<15}")
        print(f"  {'Context Precision':<25} {avg_prec:<10.4f} {'≥ 0.70':<10} "
              f"{'✓' if avg_prec >= 0.70 else '✗':<15}")
        print(f"  {'Context Recall':<25} {avg_recall:<10.4f} {'≥ 0.75':<10} "
              f"{'✓' if avg_recall >= 0.75 else '✗':<15}")
        print("  " + "-" * 66)
        print(f"  Latence moyenne:     {avg_latency:.0f} ms")
    else:
        print("  ⚠ Métriques RAGAS NON DISPONIBLES")
        print("  Pour exécuter l'évaluation complète :")
        print("    1. Installez Ollama : https://ollama.com/download")
        print("    2. Téléchargez le modèle : ollama pull mistral")
        print("    3. Lancez : ollama serve")
        print("    4. Relancez : python -m app.evaluation.eval_pipeline")
        print()

        # Statistiques retrieval uniquement
        contexts = [r.contexts for r in valid_results if r.contexts]
        if contexts:
            avg_nb_contexts = sum(len(c) for c in contexts) / len(contexts)
            avg_scores = [
                c[0]["score"]
                for c in contexts if c
            ]
            if avg_scores:
                avg_top1 = sum(avg_scores) / len(avg_scores)
                print("  STATISTIQUES DE RETRIEVAL UNIQUEMENT :")
                print(f"    Nb contextes moyen:      {avg_nb_contexts:.1f}")
                print(f"    Score top-1 moyen:       {avg_top1:.4f}")

    print()
    if valid_results:
        errors = [r for r in results if r.error is not None]
        if errors:
            print("  ERREURS RENCONTRÉES :")
            for e in errors:
                print(f"    - [{e.question[:60]}...] {e.error}")
            print()

    print("=" * 70)
    print(f"  Rapport détaillé : {REPORT_PATH}")
    print("=" * 70)
    print()


# ===========================================================================
# 6. POINT D'ENTRÉE
# ===========================================================================

def main():
    """Point d'entrée principal du pipeline d'évaluation."""
    logger.info("Pipeline d'évaluation RAGAS — Démarrage")

    # Vérification ChromaDB
    from app.indexing.vector_store import ChromaVectorStore
    try:
        vs = ChromaVectorStore(
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION,
        )
        vs.initialize_collection()
        doc_count = vs.count()
        if doc_count == 0:
            logger.warning("Collection ChromaDB vide. "
                          "Indexez d'abord des documents via la page 'Files Indexation' de l'interface.")
        else:
            logger.info("ChromaDB: %d documents indexés", doc_count)
    except Exception as e:
        logger.error("Erreur ChromaDB: %s", e)

    # Exécution de l'évaluation
    results = run_evaluation()

    # Export et rapport
    export_results(results)
    print_summary(results)

    logger.info("Pipeline d'évaluation terminé.")


if __name__ == "__main__":
    main()
