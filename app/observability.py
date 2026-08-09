"""
Per-query observability: logs latency, scores, and source files to SQLite.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("observability")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DB_PATH = Path(os.getenv("QUERY_LOG_DB", str(PROJECT_ROOT / "data" / "query_logs.db")))


class QueryLogger:
    """Logs each query to SQLite for trend analysis."""

    def __init__(self, db_path: str | Path = LOG_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    question_hash TEXT NOT NULL,
                    top1_score REAL DEFAULT 0.0,
                    mean_score REAL DEFAULT 0.0,
                    num_chunks_returned INTEGER DEFAULT 0,
                    llm_latency_ms REAL DEFAULT 0.0,
                    total_latency_ms REAL DEFAULT 0.0,
                    model_used TEXT DEFAULT '',
                    source_files TEXT DEFAULT '[]'
                )
            """)
            conn.commit()

    def log(
        self,
        question: str,
        top1_score: float,
        mean_score: float,
        num_chunks: int,
        llm_latency_ms: float,
        total_latency_ms: float,
        model_used: str = "",
        source_files: list[str] | None = None,
    ) -> None:
        """Log a single query event."""
        import json
        q_hash = hashlib.sha256(question.encode()).hexdigest()[:16]
        sources = json.dumps(source_files or [], ensure_ascii=False)

        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO query_logs
                       (timestamp, question_hash, top1_score, mean_score,
                        num_chunks_returned, llm_latency_ms, total_latency_ms,
                        model_used, source_files)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (time.time(), q_hash, top1_score, mean_score,
                     num_chunks, llm_latency_ms, total_latency_ms,
                     model_used, sources),
                )
                conn.commit()
        except Exception as e:
            logger.warning("Failed to log query: %s", e)

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get the most recent query logs."""
        import json
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for r in rows:
            entry = dict(r)
            entry["source_files"] = json.loads(entry.get("source_files", "[]"))
            results.append(entry)
        return results

    def get_stats(self, limit: int = 50) -> dict[str, Any]:
        """Get aggregated stats for recent queries."""
        recent = self.get_recent(limit)
        if not recent:
            return {
                "avg_top1_score": 0.0,
                "avg_mean_score": 0.0,
                "avg_latency_ms": 0.0,
                "query_count": 0,
                "top_sources": [],
            }

        avg_top1 = sum(r["top1_score"] for r in recent) / len(recent)
        avg_mean = sum(r["mean_score"] for r in recent) / len(recent)
        avg_latency = sum(r["total_latency_ms"] for r in recent) / len(recent)

        # Count source file usage
        source_counts: dict[str, int] = {}
        for r in recent:
            for src in r.get("source_files", []):
                source_counts[src] = source_counts.get(src, 0) + 1
        top_sources = sorted(source_counts.items(), key=lambda x: -x[1])[:10]

        return {
            "avg_top1_score": round(avg_top1, 4),
            "avg_mean_score": round(avg_mean, 4),
            "avg_latency_ms": round(avg_latency, 1),
            "query_count": len(recent),
            "top_sources": [{"source": s, "count": c} for s, c in top_sources],
        }
