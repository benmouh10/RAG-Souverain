"""Performance page — observability metrics and score trends."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.app_ui import _load_observability


def render_performance_page() -> None:
    QueryLogger = _load_observability()
    query_logger = QueryLogger()

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.header("Performance")
    st.caption("Track retrieval quality, latency, and source usage over time.")
    st.markdown("</div>", unsafe_allow_html=True)

    stats = query_logger.get_stats(limit=50)

    if stats["query_count"] == 0:
        st.info("No queries logged yet. Ask a question in the Chat RAG page to start tracking.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Top-1 Cosine", f"{stats['avg_top1_score']:.3f}")
    col2.metric("Avg Mean Score", f"{stats['avg_mean_score']:.3f}")
    col3.metric("Avg Latency", f"{stats['avg_latency_ms']:.0f} ms")
    col4.metric("Total Queries", stats["query_count"])

    recent = query_logger.get_recent(limit=50)
    if recent:
        recent.reverse()  # chronological order
        timestamps = [r["timestamp"] for r in recent]
        top1_scores = [r["top1_score"] for r in recent]
        latencies = [r["total_latency_ms"] for r in recent]

        st.subheader("Score Trend (last 50 queries)")
        score_df = pd.DataFrame({"Top-1 Score": top1_scores}, index=timestamps)
        if len(score_df) == 1:
            st.caption("Only one query logged so far — single data point shown below.")
            st.metric("Latest Top-1 Cosine Score", f"{top1_scores[0]:.3f}")
            chart_df = pd.DataFrame({"Top-1 Score": [top1_scores[0]]})
            st.line_chart(chart_df, use_container_width=True)
        else:
            st.line_chart(score_df, use_container_width=True)

        st.subheader("Latency Trend (last 50 queries)")
        latency_df = pd.DataFrame({"Latency (ms)": latencies}, index=timestamps)
        st.line_chart(latency_df, use_container_width=True)

    if stats["top_sources"]:
        st.subheader("Most Queried Sources")
        for src in stats["top_sources"]:
            st.markdown(f"- **{src['source']}** — {src['count']} queries")
