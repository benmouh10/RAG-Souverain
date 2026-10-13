"""Tests for the structured SQLite retriever (app/core/structured_db.py)."""

from __future__ import annotations

import sqlite3

import pytest

from app.core.structured_db import StructuredDBRetriever, _tokens


@pytest.fixture
def patient_db(tmp_path):
    db = tmp_path / "patients.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            nom TEXT,
            hopital VARCHAR(80),
            medecin TEXT
        );
        INSERT INTO patients VALUES
            (1, 'BEN MOUH', 'Hôtel-Dieu', 'Dr Dupont'),
            (2, 'Martin',  'Hôpital Sainte-Justine', 'Dr Tremblay');
        CREATE TABLE visites (
            visite_id INTEGER PRIMARY KEY,
            patient_id INTEGER,
            motif TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        );
        INSERT INTO visites VALUES (1, 1, 'Contrôle de routine');
        """
    )
    conn.close()
    return db


class TestAccentInsensitiveRetrieval:
    def test_unaccented_query_finds_accented_content(self, patient_db):
        retriever = StructuredDBRetriever(patient_db)
        rows = retriever.retrieve("hopital hotel dieu")
        assert rows, "unaccented tokens should match accented column values"

    def test_accented_query_finds_accented_content(self, patient_db):
        retriever = StructuredDBRetriever(patient_db)
        rows = retriever.retrieve("hôpital Hôtel-Dieu")
        assert rows

    def test_varchar_column_is_searched(self, patient_db):
        retriever = StructuredDBRetriever(patient_db)
        rows = retriever.retrieve("sainte justine")
        assert any("Hôpital Sainte-Justine" in r["content"] for r in rows)

    def test_tokenization_strips_accents(self):
        assert _tokens("hôpital") == ["hopital"]


class TestStructuredRetrievalBasics:
    def test_returns_record_with_metadata(self, patient_db):
        retriever = StructuredDBRetriever(patient_db)
        rows = retriever.retrieve("Ben Mouh")
        assert rows
        r = rows[0]
        assert r["source_file"] == patient_db.name
        assert r["table_name"] == "patients"
        assert r["record"]["nom"] == "BEN MOUH"

    def test_related_rows_expand_foreign_keys(self, patient_db):
        retriever = StructuredDBRetriever(patient_db)
        rows = retriever.retrieve("Ben Mouh")
        tables = {r["table_name"] for r in rows}
        assert "visites" in tables, "FK child rows should be included"

    def test_no_match_returns_empty(self, patient_db):
        retriever = StructuredDBRetriever(patient_db)
        assert retriever.retrieve("xylophone extraterrestre") == []

    def test_missing_db_returns_empty(self, tmp_path):
        retriever = StructuredDBRetriever(tmp_path / "absent.db")
        assert retriever.retrieve("Ben Mouh") == []