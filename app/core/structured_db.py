"""Structured SQLite retrieval for RAG.

The vector store remains useful for semantic documents, but database questions
need exact relational lookup. This module performs read-only SQLite retrieval
and expands foreign-key relationships so patient/doctor/lab/medication and
project/task questions do not depend on embedding similarity.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "what", "who", "where", "when", "which", "is", "are", "the", "a", "an",
    "of", "for", "to", "and", "or", "in", "on", "with", "about", "does", "do",
    "did", "has", "have", "had", "how", "many", "much", "my", "me", "tell",
    "what's", "whats", "information", "about", "give", "show", "please",
    "quel", "quelle", "quels", "quelles", "qui", "que", "quoi", "est", "sont",
    "de", "du", "des", "la", "le", "les", "un", "une", "pour", "dans", "sur",
    "avec", "et", "ou", "ce", "cette", "ces", "combien", "donne", "montre",
}


def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _tokens(q: str) -> list[str]:
    return [t for t in re.findall(r"[\wÀ-ÿ]+", _norm(q)) if len(t) >= 2 and t not in _STOPWORDS]


class StructuredDBRetriever:
    """Read-only, schema-aware SQLite retriever."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        # SQLite URI mode keeps the RAG from accidentally modifying the source DB.
        uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        # Accent-insensitive matching: normalize values and patterns on the fly so
        # "hôpital" and "hopital" both find rows stored with either spelling.
        conn.create_function("unaccent", 1, lambda v: _norm(v) if v is not None else None)
        return conn

    def _tables(self, conn: sqlite3.Connection) -> list[str]:
        return [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]

    def _columns(self, conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]

    def _foreign_keys(self, conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
        return [dict(r) for r in conn.execute(f'PRAGMA foreign_key_list("{table}")').fetchall()]

    def _safe_identifier(self, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError(f"Unsafe SQLite identifier: {value}")
        return value

    def _find_matching_rows(self, conn: sqlite3.Connection, query: str) -> list[tuple[str, dict[str, Any]]]:
        tokens = _tokens(query)
        if not tokens:
            return []
        matches: list[tuple[str, dict[str, Any]]] = []
        for table in self._tables(conn):
            cols = self._columns(conn, table)
            text_cols = [c["name"] for c in cols if str(c["type"]).upper().split("(")[0] in ("TEXT", "CHAR", "VARCHAR", "CLOB")]
            if not text_cols:
                continue
            # Use parameterized LIKE predicates. Requiring all query identity tokens
            # avoids returning unrelated people merely sharing one common word.
            clauses = []
            params: list[str] = []
            for col in text_cols:
                for token in tokens:
                    clauses.append(f'unaccent(CAST("{self._safe_identifier(col)}" AS TEXT)) LIKE ?')
                    params.append(f"%{token}%")
            if not clauses:
                continue
            # Any token in any text field is useful for entity discovery; scoring
            # below favors rows matching more tokens and exact names.
            sql = f'SELECT * FROM "{self._safe_identifier(table)}" WHERE ' + " OR ".join(clauses) + " LIMIT 100"
            for row in conn.execute(sql, params).fetchall():
                d = dict(row)
                hay = _norm(" ".join(str(v) for v in d.values() if v is not None))
                score = sum(1 for t in tokens if t in hay)
                if score > 0:
                    matches.append((table, d))
        # Keep the strongest entity matches first and avoid flooding context.
        matches.sort(key=lambda x: sum(1 for t in tokens if t in _norm(" ".join(str(v) for v in x[1].values()))), reverse=True)
        return matches[:12]

    def _related_rows(self, conn: sqlite3.Connection, seed: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any], str]]:
        """Expand one hop in both FK directions and label the relationship."""
        result: list[tuple[str, dict[str, Any], str]] = []
        seen: set[tuple[str, str, str]] = set()
        tables = self._tables(conn)
        for table, row in seed:
            cols = self._columns(conn, table)
            pk_cols = [c["name"] for c in cols if c["pk"]]
            for fk in self._foreign_keys(conn, table):
                local_col, parent_table, parent_col = fk["from"], fk["table"], fk["to"]
                value = row.get(local_col)
                if value is None:
                    continue
                parent = conn.execute(
                    f'SELECT * FROM "{self._safe_identifier(parent_table)}" WHERE "{self._safe_identifier(parent_col)}" = ? LIMIT 5',
                    (value,),
                ).fetchall()
                for prow in parent:
                    key = (parent_table, parent_col, str(prow[parent_col]))
                    if key not in seen:
                        seen.add(key)
                        result.append((parent_table, dict(prow), f"{table}.{local_col} -> {parent_table}.{parent_col}"))
            # Reverse lookup: find child rows pointing at this row's primary key.
            if pk_cols:
                pk = pk_cols[0]
                value = row.get(pk)
                if value is not None:
                    for child in tables:
                        for fk in self._foreign_keys(conn, child):
                            if fk["table"] != table or fk["to"] != pk:
                                continue
                            child_col = fk["from"]
                            rows = conn.execute(
                                f'SELECT * FROM "{self._safe_identifier(child)}" WHERE "{self._safe_identifier(child_col)}" = ? LIMIT 20',
                                (value,),
                            ).fetchall()
                            for crow in rows:
                                key = (child, child_col, str(crow[child_col]))
                                # Don't suppress distinct child records with same FK.
                                unique = (child, str(dict(crow)))
                                if unique not in seen:
                                    seen.add(unique)
                                    result.append((child, dict(crow), f"{child}.{child_col} -> {table}.{pk}"))
        return result[:60]

    def retrieve(self, question: str, max_seed: int = 8) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        try:
            with closing(self._connect()) as conn:
                seeds = self._find_matching_rows(conn, question)[:max_seed]
                if not seeds:
                    return []
                related = self._related_rows(conn, seeds)
                all_rows: list[tuple[str, dict[str, Any], str]] = [
                    (t, r, "direct match") for t, r in seeds
                ] + related
                # Deduplicate rows while retaining relation information.
                seen: set[tuple[str, str]] = set()
                output: list[dict[str, Any]] = []
                for table, row, relation in all_rows:
                    key = (table, repr(sorted(row.items())))
                    if key in seen:
                        continue
                    seen.add(key)
                    lines = [
                        f"[STRUCTURED DATABASE: {self.db_path.name}]",
                        f"Table: {table}",
                        f"Relationship: {relation}",
                        "Data:",
                    ]
                    for k, v in row.items():
                        if v is not None and str(v).strip() != "":
                            lines.append(f"  - {k}: {v}")
                    output.append({
                        "content": "\n".join(lines),
                        "source_file": self.db_path.name,
                        "table_name": table,
                        "record": row,
                    })
                return output[:60]
        except Exception:
            return []
