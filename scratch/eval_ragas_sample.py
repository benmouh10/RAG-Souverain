"""
Évaluation RAGAS sur un échantillon de 10 questions.
Lit les réponses déjà générées depuis le CSV, récupère les contextes,
et calcule les 4 métriques RAGAS.
"""
import csv
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("eval_sample")

def main():
    csv_path = PROJECT_ROOT / "data" / "evaluation_report.csv"
    if not csv_path.exists():
        logger.error("Fichier non trouve: %s", csv_path)
        return

    # 1. Initialiser le RAG Engine pour recuperer les contextes
    logger.info("Initialisation du moteur RAG...")
    from app.core.rag_engine import RAGEngine
    rag = RAGEngine()

    # 2. Configurer RAGAS
    logger.info("Configuration de RAGAS...")
    from app.evaluation.eval_pipeline import setup_ragas_evaluator, RAGAS_TIMEOUT_SECONDS, _run_with_timeout
    
    ragas_result = setup_ragas_evaluator()
    if ragas_result is None:
        logger.error("RAGAS n'a pas pu etre initialise.")
        return

    # ragas_result is a list of 4 metrics: [faithfulness, answer_relevancy, context_precision, context_recall]
    metrics = ragas_result

    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Get the LLM and embeddings from the metrics (they were already set by setup_ragas_evaluator)
    evaluator_llm = metrics[0].llm
    evaluator_embeddings = metrics[1].embeddings

    from datasets import Dataset

    # 3. Lire les 10 premieres lignes du CSV
    eval_data = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:
                break
            eval_data.append(row)

    logger.info("%d questions chargees depuis le CSV.", len(eval_data))

    all_scores = []

    for i, row in enumerate(eval_data, start=1):
        q = row["question"]
        ref = row["reference"]
        ans = row["response"]

        logger.info("[%d/10] Question: %s", i, q[:80])

        # Recuperer les contextes via le retriever (rapide, pas de LLM)
        search_results = rag.retriever.retrieve(
            q,
            top_k=rag.top_k,
            min_score=rag.min_score
        )
        contexts = [res.content for res in search_results]

        if not contexts:
            logger.warning("  -> Aucun contexte retrouve.")
            continue

        dataset_dict = {
            "question": [q],
            "answer": [ans],
            "contexts": [contexts],
            "ground_truth": [ref]
        }
        dataset = Dataset.from_dict(dataset_dict)

        logger.info("  -> Calcul des metriques RAGAS (%d contextes)...", len(contexts))

        def _score():
            return evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                raise_exceptions=False
            )

        try:
            scored = _run_with_timeout(_score, timeout=RAGAS_TIMEOUT_SECONDS)
        except Exception as e:
            logger.error("  -> Erreur RAGAS: %s", e)
            continue

        if scored is not None:
            df = scored.to_pandas()
            if not df.empty:
                first_row = df.iloc[0]
                f_score = float(first_row.get("faithfulness", 0.0))
                ar_score = float(first_row.get("answer_relevancy", 0.0))
                cp_score = float(first_row.get("context_precision", 0.0))
                cr_score = float(first_row.get("context_recall", 0.0))
                logger.info("  -> Faith=%.3f | AnsRel=%.3f | CtxPrec=%.3f | CtxRec=%.3f",
                            f_score, ar_score, cp_score, cr_score)
                all_scores.append({
                    "question": q,
                    "faithfulness": f_score,
                    "answer_relevancy": ar_score,
                    "context_precision": cp_score,
                    "context_recall": cr_score
                })
        else:
            logger.error("  -> Timeout RAGAS.")

        time.sleep(0.5)

    # Summary
    if all_scores:
        n = len(all_scores)
        avg_f = sum(s["faithfulness"] for s in all_scores) / n
        avg_ar = sum(s["answer_relevancy"] for s in all_scores) / n
        avg_cp = sum(s["context_precision"] for s in all_scores) / n
        avg_cr = sum(s["context_recall"] for s in all_scores) / n

        print("\n" + "=" * 60)
        print("  RAGAS EVALUATION SUMMARY (10 questions sample)")
        print("=" * 60)
        print(f"  Questions evaluated:    {n}")
        print(f"  Avg Faithfulness:       {avg_f:.3f}")
        print(f"  Avg Answer Relevancy:   {avg_ar:.3f}")
        print(f"  Avg Context Precision:  {avg_cp:.3f}")
        print(f"  Avg Context Recall:     {avg_cr:.3f}")
        print("=" * 60)

        # Save to CSV
        out_path = PROJECT_ROOT / "data" / "ragas_sample_scores.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"])
            writer.writeheader()
            writer.writerows(all_scores)
            # Write averages row
            writer.writerow({
                "question": "AVERAGE",
                "faithfulness": f"{avg_f:.3f}",
                "answer_relevancy": f"{avg_ar:.3f}",
                "context_precision": f"{avg_cp:.3f}",
                "context_recall": f"{avg_cr:.3f}"
            })
        logger.info("Scores sauvegardes dans: %s", out_path)
    else:
        print("Aucun score RAGAS n'a pu etre calcule.")

    logger.info("Evaluation terminee.")

if __name__ == "__main__":
    main()
