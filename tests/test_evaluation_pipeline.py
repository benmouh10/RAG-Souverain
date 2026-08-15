"""Tests du pipeline d'évaluation (Sprint 7 / M4).

- Unitaires  : export du rapport CSV (schéma, valeurs formatées) — sans réseau.
- Intégration : boucle d'évaluation sur un sous-ensemble du dataset doré,
  export vers ``data/evaluation_report.csv`` — le fichier lu par la page
  « Evaluation & Audit » (report section).

Exécution (Ollama + modèle d'embeddings + collection ChromaDB requis) :
    python -m pytest tests/test_evaluation_pipeline.py -m integration
    EVALUATION_LIMIT=5 python -m pytest tests/test_evaluation_pipeline.py -m integration

La variable d'environnement ``EVALUATION_LIMIT`` borne le nombre de questions
traitées (défaut : l'intégralité du dataset doré, soit 30 questions).
"""

from __future__ import annotations

import os

import pytest

from app.evaluation.eval_pipeline import (
    GOLD_DATASET,
    EvalResult,
    REPORT_PATH,
    export_results,
    run_evaluation,
)

EXPECTED_COLUMNS = [
    "question", "reference", "response",
    "faithfulness", "answer_relevancy",
    "context_precision", "context_recall",
    "latency_ms", "error", "nb_contexts",
    "top1_score", "top1_source",
]


def _sample_result(
    question: str,
    *,
    metrics: bool = True,
    error: str | None = None,
    nb_contexts: int = 3,
) -> EvalResult:
    contexts = [
        {"content": f"contexte {i}", "score": 0.9 - i * 0.1, "source": "stage_si_RAG.pdf", "page": i + 1}
        for i in range(nb_contexts)
    ]
    result = EvalResult(
        question=question,
        reference="référence de test",
        response="réponse générée par le RAG.",
        contexts=contexts,
        latency_ms=1234.5,
    )
    if metrics and error is None:
        result.faithfulness = 0.9
        result.answer_relevancy = 0.8
        result.context_precision = 0.7
        result.context_recall = 0.75
    if error is not None:
        result.error = error
    return result


# ===========================================================================
# Unitaires — export du rapport (aucun réseau requis)
# ===========================================================================

class TestExportResults:
    def test_export_schema_and_values(self, tmp_path):
        out = tmp_path / "report.csv"
        results = [
            _sample_result("Question A"),
            _sample_result("Question B", error="Ollama timeout"),
        ]
        export_results(results, out)

        import pandas as pd

        df = pd.read_csv(out, dtype=str, keep_default_na=False)
        assert list(df.columns) == EXPECTED_COLUMNS
        assert len(df) == 2

        ok = df.iloc[0]
        assert ok["question"] == "Question A"
        assert ok["response"] == "réponse générée par le RAG."
        assert ok["faithfulness"] == "0.9000"
        assert ok["answer_relevancy"] == "0.8000"
        assert ok["top1_source"] == "stage_si_RAG.pdf"
        assert ok["top1_score"] == "0.9000"
        assert ok["nb_contexts"] == "3"

        err = df.iloc[1]
        assert err["error"] == "Ollama timeout"
        assert err["faithfulness"] == "N/A"

    def test_export_empty_list(self, tmp_path):
        out = tmp_path / "empty.csv"
        export_results([], out)
        import pandas as pd

        df = pd.read_csv(out)
        assert list(df.columns) == EXPECTED_COLUMNS
        assert df.empty


# ===========================================================================
# Intégration — boucle d'évaluation réelle → rapport lisible par la page
# ===========================================================================

@pytest.mark.integration
def test_evaluation_roundtrip(monkeypatch):
    """Lance l'évaluation sur un sous-ensemble, exporte le rapport vers
    ``data/evaluation_report.csv`` et vérifie qu'il est exploitable par la
    page Evaluation & Audit (schéma + réponses générées)."""
    import pandas as pd
    from pathlib import Path

    limit = int(os.getenv("EVALUATION_LIMIT", str(len(GOLD_DATASET))))
    limit = max(1, min(limit, len(GOLD_DATASET)))
    subset = GOLD_DATASET[:limit]
    monkeypatch.setattr("app.evaluation.eval_pipeline.GOLD_DATASET", subset)

    results = run_evaluation(timeout_seconds=120)
    export_results(results)

    report = Path(REPORT_PATH)
    assert report.exists(), "Le rapport n'a pas été généré."

    df = pd.read_csv(report)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == limit

    answered = df[df["response"].notna() & (df["response"].str.len() > 0)]
    assert len(answered) >= 1, "Aucune réponse générée par le RAG."
    assert answered["nb_contexts"].notna().any(), "Aucun contexte de retrieval collecté."
