"""Offline tests for app.core.sql_query_engine.

Runs without Ollama or Streamlit: a real temporary SQLite file stands in for
an uploaded ``.db``, and the LLM call in the full pipeline test is a stub
returning a fixed query (SQL generation itself needs a live model; only the
plumbing around it — schema introspection, safety validation, read-only
execution, HTML rendering, database selection — is unit-tested here).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core import sql_query_engine as sqe  # noqa: E402


def _make_sample_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department_id INTEGER,
            salary REAL,
            FOREIGN KEY(department_id) REFERENCES departments(id)
        );
        INSERT INTO departments (id, name) VALUES (1, 'Engineering'), (2, 'Sales');
        INSERT INTO employees (id, name, department_id, salary) VALUES
            (1, 'Alice', 1, 90000),
            (2, 'Bob', 1, 85000),
            (3, 'Carol', 2, 60000);
        """
    )
    conn.commit()
    conn.close()


class FakeLLM:
    """Stub matching the OllamaLLM.generate_sql / generate_routed_sql interface."""

    def __init__(self, sql: str = "", routed_response: str = "") -> None:
        self.sql = sql
        self.routed_response = routed_response
        self.circuit_open = False

    def generate_sql(self, question: str, schema_text: str) -> str:
        return self.sql

    def generate_routed_sql(self, question: str, schemas: dict) -> str:
        return self.routed_response


def test_is_structured_query_detects_aggregates():
    assert sqe.is_structured_query("How many employees are in Engineering?")
    assert sqe.is_structured_query("Combien d'employés travaillent aux Ventes ?")
    assert sqe.is_structured_query("What is the average salary per department?")
    assert sqe.is_structured_query("Quelle est la moyenne des salaires ?")
    assert sqe.is_structured_query("List the top 5 highest paid employees")
    assert not sqe.is_structured_query("Who is mohamed ben mouh?")
    assert not sqe.is_structured_query("What is the RAG system?")
    assert not sqe.is_structured_query("")


def test_discover_db_files(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    documents = tmp_path / "documents"
    uploads.mkdir()
    documents.mkdir()
    (uploads / "company.db").write_bytes(b"")
    (documents / "other.db").write_bytes(b"")

    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads, documents))
    found = sqe.discover_db_files()
    assert set(found.keys()) == {"company.db", "other.db"}
    assert found["company.db"] == uploads / "company.db"


def test_inspect_schema_and_format(tmp_path):
    db_path = tmp_path / "company.db"
    _make_sample_db(db_path)

    tables = sqe.inspect_schema(db_path)
    names = {t.name for t in tables}
    assert names == {"departments", "employees"}

    emp = next(t for t in tables if t.name == "employees")
    assert emp.row_count == 3
    assert ("department_id", "departments", "id") in emp.foreign_keys

    schema_text = sqe.format_schema_for_prompt(tables)
    assert "Table: employees" in schema_text
    assert "-> department_id references departments.id" in schema_text


def test_validate_select_only_accepts_clean_select():
    sql = sqe.validate_select_only("SELECT COUNT(*) FROM employees")
    assert sql.upper().startswith("SELECT")
    assert "LIMIT" in sql.upper()  # auto-appended since absent


def test_validate_select_only_accepts_cte_and_keeps_existing_limit():
    sql = sqe.validate_select_only(
        "WITH t AS (SELECT * FROM employees) SELECT * FROM t LIMIT 10"
    )
    assert sql.upper().count("LIMIT") == 1


def test_validate_select_only_strips_markdown_fences():
    sql = sqe.validate_select_only("```sql\nSELECT * FROM employees\n```")
    assert sql.upper().startswith("SELECT")


def test_validate_select_only_rejects_mutations():
    for bad in [
        "DELETE FROM employees",
        "DROP TABLE employees",
        "UPDATE employees SET salary = 0",
        "INSERT INTO employees VALUES (4, 'x', 1, 1)",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA writable_schema=1",
    ]:
        try:
            sqe.validate_select_only(bad)
            assert False, f"expected rejection for: {bad}"
        except sqe.SQLSafetyError:
            pass


def test_validate_select_only_rejects_stacked_statements():
    try:
        sqe.validate_select_only("SELECT 1; DROP TABLE employees;")
        assert False, "expected rejection for stacked statements"
    except sqe.SQLSafetyError:
        pass


def test_validate_select_only_rejects_comments():
    try:
        sqe.validate_select_only("SELECT * FROM employees -- sneaky")
        assert False, "expected rejection for SQL comments"
    except sqe.SQLSafetyError:
        pass


def test_execute_readonly_sql_returns_exact_rows(tmp_path):
    db_path = tmp_path / "company.db"
    _make_sample_db(db_path)

    columns, rows = sqe.execute_readonly_sql(
        db_path,
        "SELECT department_id, COUNT(*) AS n FROM employees GROUP BY department_id ORDER BY department_id",
    )
    assert columns == ["department_id", "n"]
    assert rows == [(1, 2), (2, 1)]


def test_execute_readonly_sql_cannot_mutate(tmp_path):
    """Defense in depth: even if a mutating statement reached execution,
    the read-only connection must refuse to write."""
    db_path = tmp_path / "company.db"
    _make_sample_db(db_path)

    conn = sqe._connect_readonly(db_path)
    try:
        try:
            conn.execute("DELETE FROM employees")
            assert False, "read-only connection unexpectedly allowed a write"
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()

    # Data untouched.
    conn2 = sqlite3.connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    conn2.close()
    assert count == 3


def test_render_result_html_scalar():
    html_out = sqe.render_result_html(["n"], [(3,)])
    assert "<strong>Answer:</strong> 3" in html_out


def test_render_result_html_table():
    html_out = sqe.render_result_html(["id", "name"], [(1, "Alice"), (2, "Bob")])
    assert "<table" in html_out and "Alice" in html_out and "Bob" in html_out


def test_render_result_html_empty():
    html_out = sqe.render_result_html(["id"], [])
    assert "no matching rows" in html_out


def test_full_pipeline_with_stub_llm(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    db_path = uploads / "company.db"
    _make_sample_db(db_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    assert engine.has_databases()
    assert engine.list_databases() == ["company.db"]

    llm = FakeLLM("SELECT department_id, COUNT(*) FROM employees GROUP BY department_id")
    result = engine.answer("How many employees per department?", llm)

    assert result.db_file == "company.db"
    assert result.rows == [(1, 2), (2, 1)]
    assert "row(s) found" in result.answer_html


def test_full_pipeline_rejects_malicious_llm_output(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    db_path = uploads / "company.db"
    _make_sample_db(db_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    llm = FakeLLM("DELETE FROM employees; SELECT 1")
    try:
        engine.answer("Delete everyone", llm)
        assert False, "expected SQLSafetyError to propagate"
    except sqe.SQLSafetyError:
        pass

    # Data still intact.
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    conn.close()
    assert count == 3


def test_full_pipeline_no_query_response(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    db_path = uploads / "company.db"
    _make_sample_db(db_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    llm = FakeLLM("NO_QUERY")
    try:
        engine.answer("What's the weather like?", llm)
        assert False, "expected SQLGenerationError"
    except sqe.SQLGenerationError:
        pass


# ---------------------------------------------------------------------------
# Multi-database routing
# ---------------------------------------------------------------------------
def _make_sales_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE deals (
            id INTEGER PRIMARY KEY,
            client TEXT NOT NULL,
            amount REAL NOT NULL
        );
        INSERT INTO deals (id, client, amount) VALUES
            (1, 'Acme', 12000),
            (2, 'Globex', 8000);
        """
    )
    conn.commit()
    conn.close()


def _two_db_uploads(tmp_path: Path) -> Path:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    _make_sample_db(uploads / "hr.db")
    _make_sales_db(uploads / "sales.db")
    return uploads


def test_parse_routed_response_valid():
    db, sql = sqe.parse_routed_response("DATABASE: sales.db\nSQL: SELECT AVG(amount) FROM deals")
    assert db == "sales.db"
    assert sql == "SELECT AVG(amount) FROM deals"


def test_parse_routed_response_no_query():
    db, sql = sqe.parse_routed_response("NO_QUERY")
    assert db is None and sql == ""


def test_parse_routed_response_malformed():
    db, sql = sqe.parse_routed_response("I think it's sales.db, query: SELECT * FROM deals")
    assert db is None and sql == ""


def test_multi_db_ambiguous_question_routes_via_llm(tmp_path, monkeypatch):
    uploads = _two_db_uploads(tmp_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    assert set(engine.list_databases()) == {"hr.db", "sales.db"}

    llm = FakeLLM(routed_response="DATABASE: sales.db\nSQL: SELECT AVG(amount) FROM deals")
    result = engine.answer("What is the average deal size?", llm)

    assert result.db_file == "sales.db"
    assert result.rows == [(10000.0,)]


def test_multi_db_explicit_mention_skips_routing(tmp_path, monkeypatch):
    """A filename literally present in the question (with or without an
    '@' prefix — plain substring matching) resolves directly without
    calling generate_routed_sql at all."""
    uploads = _two_db_uploads(tmp_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()

    class RoutingForbiddenLLM(FakeLLM):
        def generate_routed_sql(self, question, schemas):
            raise AssertionError("routing should not be triggered when a file is named")

    llm = RoutingForbiddenLLM(sql="SELECT COUNT(*) FROM employees")

    def fake_generate_sql(question, schema_text):
        assert "employees" in schema_text
        return "SELECT COUNT(*) FROM employees"

    llm.generate_sql = fake_generate_sql
    result = engine.answer("@hr.db how many employees do we have?", llm)
    assert result.db_file == "hr.db"
    assert result.rows == [(3,)]


def test_multi_db_file_hint_skips_routing(tmp_path, monkeypatch):
    uploads = _two_db_uploads(tmp_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()

    class RoutingForbiddenLLM(FakeLLM):
        def generate_routed_sql(self, question, schemas):
            raise AssertionError("routing should not be triggered when file_hint is given")

    llm = RoutingForbiddenLLM(sql="SELECT AVG(amount) FROM deals")
    result = engine.answer("what's the average?", llm, file_hint="sales.db")
    assert result.db_file == "sales.db"
    assert result.rows == [(10000.0,)]


def test_multi_db_routing_rejects_unknown_database_name(tmp_path, monkeypatch):
    uploads = _two_db_uploads(tmp_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    llm = FakeLLM(routed_response="DATABASE: made_up.db\nSQL: SELECT 1")
    try:
        engine.answer("some ambiguous question", llm)
        assert False, "expected SQLGenerationError for a hallucinated database name"
    except sqe.SQLGenerationError:
        pass


def test_multi_db_routing_no_query(tmp_path, monkeypatch):
    uploads = _two_db_uploads(tmp_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    llm = FakeLLM(routed_response="NO_QUERY")
    try:
        engine.answer("what's the meaning of life?", llm)
        assert False, "expected SQLGenerationError"
    except sqe.SQLGenerationError:
        pass


def test_multi_db_routing_rejects_malicious_sql(tmp_path, monkeypatch):
    uploads = _two_db_uploads(tmp_path)
    monkeypatch.setattr(sqe, "DB_SEARCH_DIRS", (uploads,))

    engine = sqe.SQLQueryEngine()
    llm = FakeLLM(routed_response="DATABASE: hr.db\nSQL: DELETE FROM employees")
    try:
        engine.answer("wipe everything", llm)
        assert False, "expected SQLSafetyError"
    except sqe.SQLSafetyError:
        pass


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
