"""Evaluation & Audit page — RAGAS test runner with per-query timeouts and progress."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

from app.app_ui import _load_eval_helpers
from app.session import require_admin

logger = logging.getLogger("rag_ui")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _run_evaluation_background(
    progress_placeholder: st.delta_generator,
    progress_bar: st.delta_generator,
) -> None:
    export_results, run_evaluation = _load_eval_helpers()
    try:
        progress_placeholder.info("Running RAGAS evaluation pipeline...")

        def _on_progress(done: int, total: int) -> None:
            pct = min(1.0, max(0.0, done / total))
            progress_bar.progress(pct)
            progress_placeholder.info(f"Evaluating question {done}/{total}…")

        results = run_evaluation(progress_callback=_on_progress, timeout_seconds=30)
        export_results(results)
        progress_bar.progress(1.0)
        progress_placeholder.success(f"Evaluation complete — {len(results)} questions processed. Report exported to data/evaluation_report.csv")
    except ConnectionError as exc:
        progress_placeholder.error("Ollama inaccessible — evaluation requires an active LLM.")
        logger.warning("Eval failed (Ollama): %s", exc)
    except ImportError as exc:
        progress_placeholder.error(f"Missing dependency for evaluation: {exc}")
        logger.warning("Eval failed (import): %s", exc)
    except Exception as exc:
        progress_placeholder.error(f"Evaluation error: {type(exc).__name__}: {exc}")
        logger.exception("Evaluation error")
    finally:
        st.session_state["eval_running"] = False


def render_evaluation_page() -> None:
    if not require_admin():
        return

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Evaluation & Audit")
    st.caption("Measure the fidelity and relevance of RAG-generated answers against your reference documents.")
    st.markdown("</div>", unsafe_allow_html=True)

    report_path = PROJECT_ROOT / "data" / "evaluation_report.csv"

    if not report_path.exists():
        st.info("No evaluation report generated yet. Run an evaluation session to compute scores.")
    else:
        try:
            df = pd.read_csv(report_path)
        except pd.errors.EmptyDataError:
            st.info("Report file is empty. Re-run an evaluation.")
            df = None
        except pd.errors.ParserError as exc:
            st.error(f"Report file is corrupted: {exc}")
            df = None
        except PermissionError:
            st.error("Cannot read report file — insufficient permissions.")
            df = None
        except Exception as exc:
            st.error(f"Cannot read CSV report: {type(exc).__name__}: {exc}")
            df = None

        if df is not None and not df.empty:
            metrics = [col for col in ("faithfulness", "answer_relevancy", "context_precision", "context_recall") if col in df.columns]
            if metrics:
                agg = df[metrics].apply(pd.to_numeric, errors="coerce").mean()

                targets = {
                    "faithfulness": 0.80,
                    "answer_relevancy": 0.70,
                    "context_precision": 0.70,
                    "context_recall": 0.75,
                }

                st.subheader("Average RAGAS metrics")
                cols = st.columns(len(agg))
                for col, (key, value) in zip(cols, agg.items()):
                    metric_value = f"{value:.4f}" if pd.notna(value) else "N/A"
                    cols_delta = float(value) - targets[key] if pd.notna(value) else None
                    col.metric(
                        key.replace("_", " ").title(),
                        metric_value,
                        delta=f"{cols_delta:+.4f}" if cols_delta is not None else None,
                        delta_color="normal",
                    )

                summary_rows = []
                for key, value in agg.items():
                    achieved = float(value) if pd.notna(value) else None
                    passed = achieved is not None and achieved >= targets[key]
                    summary_rows.append({
                        "Metric": key.replace("_", " ").title(),
                        "Target": f"≥ {targets[key]:.2f}",
                        "Achieved": f"{achieved:.4f}" if achieved is not None else "N/A",
                        "Status": "✅ Pass" if passed else "❌ Fail",
                        "Pass Rate": f"{achieved:.2%}" if achieved is not None else "N/A",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

            with st.expander("Show detailed report", expanded=False):
                st.dataframe(df.fillna("N/A"), use_container_width=True)
        elif df is not None and df.empty:
            st.info("Report is empty. Run an evaluation to populate it.")

    st.markdown("---")

    progress_placeholder = st.empty()
    progress_bar = st.progress(0.0)

    if st.button("Run local evaluation", use_container_width=True):
        if st.session_state["eval_running"]:
            st.warning("An evaluation is already in progress.")
        else:
            st.session_state["eval_running"] = True
            thread = threading.Thread(
                target=_run_evaluation_background,
                args=(progress_placeholder, progress_bar),
                daemon=True,
            )
            thread.start()
            st.info("Evaluation started in the background. A per-question timeout prevents indefinite hangs.")

    if st.button("Refresh report"):
        st.rerun()
