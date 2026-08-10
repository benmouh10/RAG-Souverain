"""Tests verrouillant les correctifs issus de la review (C1, C4, M1, M2, C5).

C1/Mo4 — versionnage global des documents (doc_versions sans user_id) + migration.
C4     — search_latest ne renvoie que les chunks de la dernière version (égalité).
M1     — garde serveur require_admin() (RBAC côté session).
M2     — health_check() + circuit breaker de OllamaLLM.
C5     — périmètre restreint du watcher (patterns ignorés, garde is_file).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
import requests

from app.ingestion.chunker import ChunkStrategy, TextChunk


class _SessionState:
    def __init__(self, data: dict) -> None:
        self._data = data

    def __getitem__(self, key: str):
        return self._data[key]

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def __setitem__(self, key: str, value) -> None:
        self._data[key] = value


class _FakeStreamlit:
    """Remplace `streamlit` (via app.session.st) pour tester session.py sans UI."""

    def __init__(self, **state) -> None:
        self.session_state = _SessionState(dict(state))
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)


class TestGlobalVersioning:
    """C1/Mo4 — le versionnage est global par filename, sans user_id."""

    def test_record_ingestion_increments_globally(self, tmp_path):
        from app.database import Database

        db = Database(tmp_path / "rag.db")
        v1 = db.record_ingestion("a.pdf", "h1", 4)
        v2 = db.record_ingestion("a.pdf", "h2", 6)

        assert v1.version == 1
        assert v2.version == 2
        assert db.get_latest_version("a.pdf").version == 2
        assert db.get_file_hash("a.pdf") == "h2"

    def test_versioning_shared_across_admins(self, tmp_path):
        from app.database import Database

        db = Database(tmp_path / "rag.db")
        db.record_ingestion("a.pdf", "h1", 4)
        # Pas de clé user_id : aucune notion de version « par utilisateur ».
        latest = db.get_latest_version("a.pdf")
        assert latest.version == 1
        assert latest.chunk_count == 4

        v = db.record_ingestion("a.pdf", "h3", 7)
        assert v.version == 2

    def test_list_documents_and_total_chunks(self, tmp_path):
        from app.database import Database

        db = Database(tmp_path / "rag.db")
        db.record_ingestion("a.pdf", "h1", 4)
        db.record_ingestion("b.txt", "hx", 3)

        docs = db.list_documents()
        assert {d.filename for d in docs} == {"a.pdf", "b.txt"}
        assert db.get_total_chunks() == 7

    def test_migrates_legacy_user_id_schema(self, tmp_path):
        from app.database import Database

        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(
            """
            CREATE TABLE doc_versions (
                doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                user_id     INTEGER NOT NULL,
                file_hash   TEXT NOT NULL,
                version     INTEGER NOT NULL DEFAULT 1,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                indexed_at  REAL NOT NULL
            );
            INSERT INTO doc_versions (filename, user_id, file_hash, version, chunk_count, indexed_at)
            VALUES ('a.pdf', 1, 'h1', 1, 3, 1000.0),
                   ('a.pdf', 2, 'h1', 1, 3, 1001.0),
                   ('b.txt', 1, 'hx', 1, 2, 1002.0);
            """
        )
        conn.commit()
        conn.close()

        db = Database(db_path)
        columns = {r[1] for r in db._connect().execute("PRAGMA table_info(doc_versions)").fetchall()}
        assert "user_id" not in columns

        docs = db.list_documents()
        assert len(docs) == 3  # lignes de l'ancien schéma préservées

        v = db.record_ingestion("a.pdf", "h3", 7)
        assert v.version == 2  # re-indexation après migration


class TestSearchLatest:
    """C4 — search_latest ne renvoie que les chunks de la version cible."""

    def _add(self, vs, filename: str, version: int, content: str) -> None:
        vs.add_chunks(
            [
                TextChunk(
                    content=content,
                    source_file=filename,
                    page_number=1,
                    chunk_index=0,
                    strategy=ChunkStrategy.SLIDING_WINDOW,
                    version=version,
                )
            ]
        )

    def test_search_latest_returns_only_current_version(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "doc.pdf", 1, "ancienne version du document.")
        self._add(vs, "doc.pdf", 2, "nouvelle version du document.")

        results = vs.search_latest("document", top_k=5)

        assert results
        assert all(r.metadata["version"] == 2 for r in results)
        # La recherche non filtrée, elle, retrouve les deux versions.
        all_results = vs.search("document", top_k=5)
        assert len(all_results) == 2

    def test_search_latest_with_explicit_max_version(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "doc.pdf", 1, "ancienne version du document.")
        self._add(vs, "doc.pdf", 2, "nouvelle version du document.")

        results = vs.search_latest("document", top_k=5, max_version=1)

        assert results
        assert all(r.metadata["version"] == 1 for r in results)

    def test_get_latest_version_all(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "doc.pdf", 1, "version une.")
        self._add(vs, "other.txt", 3, "version trois.")

        assert vs.get_latest_version_all() == 3
        assert vs.get_latest_version("doc.pdf") == 1

    def test_search_latest_on_empty_collection(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()

        assert vs.get_latest_version_all() is None
        assert vs.search_latest("anything", top_k=5) == []

    def test_delete_by_source_removes_all_versions(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "doc.pdf", 1, "ancienne version du document.")
        self._add(vs, "doc.pdf", 2, "nouvelle version du document.")

        deleted = vs.delete_by_source("doc.pdf")

        assert deleted == 2
        assert vs.count() == 0


class TestHealthCheckAndCircuitBreaker:
    """M2 — health_check() et circuit breaker de OllamaLLM."""

    def _connection_refused_get(self, url, timeout):
        raise requests.exceptions.ConnectionError("connection refused")

    def test_health_check_reports_failure(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        monkeypatch.setattr("requests.get", self._connection_refused_get)
        llm = OllamaLLM(base_url="http://localhost:1")

        assert llm.health_check() is False
        assert llm.last_health is False

    def test_health_check_success_resets_state(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        class _Resp:
            status_code = 200

        monkeypatch.setattr("requests.get", lambda url, timeout: _Resp())
        llm = OllamaLLM(base_url="http://localhost:1")

        assert llm.health_check() is True
        assert llm.last_health is True
        assert llm.circuit_open is False

    def test_circuit_opens_after_consecutive_failures(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        monkeypatch.setattr("requests.get", self._connection_refused_get)
        llm = OllamaLLM(base_url="http://localhost:1")

        for _ in range(3):
            assert llm.health_check() is False

        assert llm.circuit_open is True

    def test_circuit_closes_after_success(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        class _Resp:
            status_code = 200

        def _fake_get(url, timeout):
            if _fake_get.failures > 0:
                _fake_get.failures -= 1
                raise requests.exceptions.ConnectionError("connection refused")
            return _Resp()

        _fake_get.failures = 3
        monkeypatch.setattr("requests.get", _fake_get)
        llm = OllamaLLM(base_url="http://localhost:1")

        for _ in range(3):
            llm.health_check()
        assert llm.circuit_open is True

        assert llm.health_check() is True
        assert llm.circuit_open is False

    def test_generate_stream_short_circuits_when_open(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        def _boom(*args, **kwargs):
            raise AssertionError("aucun appel réseau ne doit être émis circuit ouvert")

        monkeypatch.setattr("requests.post", _boom)
        monkeypatch.setattr("requests.get", _boom)

        llm = OllamaLLM(base_url="http://localhost:1")
        llm._circuit_open_until = time.time() + 100

        out = list(llm.generate_stream("question", "contexte"))

        assert out
        assert "indisponible" in out[0]

    def test_generate_stream_registers_failure_on_connection_error(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        def _refused(*args, **kwargs):
            raise requests.exceptions.ConnectionError("connection refused")

        monkeypatch.setattr("requests.post", _refused)
        monkeypatch.setattr("requests.get", _refused)

        llm = OllamaLLM(base_url="http://localhost:1")
        out = list(llm.generate_stream("question", "contexte"))

        assert out and "Erreur" in out[0]
        assert llm._consecutive_failures >= 1


class TestRequireAdminGuard:
    """M1 — garde serveur require_admin() côté session."""

    def test_require_admin_allows_admin(self, monkeypatch):
        from app import session as sess

        fake = _FakeStreamlit(user_role="admin")
        monkeypatch.setattr(sess, "st", fake)

        assert sess.require_admin() is True
        assert fake.errors == []

    def test_require_admin_blocks_non_admin(self, monkeypatch):
        from app import session as sess

        fake = _FakeStreamlit(user_role="user")
        monkeypatch.setattr(sess, "st", fake)
        monkeypatch.setattr("streamlit.error", fake.error)

        assert sess.require_admin() is False
        assert fake.errors  # l'accès refusé est signalé à l'utilisateur

    def test_is_admin_reads_role_and_user_role(self, monkeypatch):
        from app import session as sess

        for key in ("user_role", "role"):
            fake = _FakeStreamlit(**{key: "admin"})
            monkeypatch.setattr(sess, "st", fake)
            assert sess.is_admin() is True

            fake_user = _FakeStreamlit(**{key: "user"})
            monkeypatch.setattr(sess, "st", fake_user)
            assert sess.is_admin() is False


class TestWatcherScope:
    """C5 — périmètre restreint du watcher."""

    def test_ignored_patterns(self, tmp_path):
        from app.ingestion.watcher import _should_ignore

        for name in (".hidden.txt", "backup.txt~", "~draft.txt", ".DS_Store", "file.swp", "file.swo"):
            assert _should_ignore(tmp_path / name), name

        assert not _should_ignore(tmp_path / "doc.txt")
        assert not _should_ignore(tmp_path / "rapport.pdf")

    def test_schedule_ingest_guards(self, tmp_path):
        from app.ingestion.watcher import WatcherEventHandler

        handler = WatcherEventHandler(conn=None)

        (tmp_path / "subdir").mkdir()
        handler._schedule_ingest(tmp_path / "subdir")            # répertoire → ignoré
        (tmp_path / "notes.md.tmp").write_text("x", encoding="utf-8")
        handler._schedule_ingest(tmp_path / "notes.md.tmp")      # extension non supportée
        (tmp_path / ".hidden.txt").write_text("x", encoding="utf-8")
        handler._schedule_ingest(tmp_path / ".hidden.txt")       # pattern ignoré (.*)
        assert handler._pending == {}

        real = tmp_path / "doc.txt"
        real.write_text("contenu", encoding="utf-8")
        handler._schedule_ingest(real)
        assert str(real) in handler._pending

    def test_scan_skips_directories_and_unsupported(self, tmp_path, monkeypatch):
        from app.ingestion.watcher import _init_hash_db, _scan_and_ingest

        (tmp_path / "subdir").mkdir()
        (tmp_path / "notes.md.tmp").write_text("x", encoding="utf-8")
        (tmp_path / "data.db").write_text("x", encoding="utf-8")
        monkeypatch.setattr("app.ingestion.watcher.WATCH_DIR", tmp_path)
        monkeypatch.setattr("app.ingestion.watcher.HASH_DB_PATH", tmp_path / "hashes.db")

        conn = _init_hash_db()
        _scan_and_ingest(conn)  # aucun .pdf/.txt/.md → aucune ingestion tentée
        conn.close()

    def test_scan_when_watch_dir_missing(self, tmp_path, monkeypatch):
        from app.ingestion.watcher import _init_hash_db, _scan_and_ingest

        monkeypatch.setattr("app.ingestion.watcher.WATCH_DIR", tmp_path / "absent")
        monkeypatch.setattr("app.ingestion.watcher.HASH_DB_PATH", tmp_path / "hashes.db")

        conn = _init_hash_db()
        _scan_and_ingest(conn)  # répertoire absent → retour immédiat
        conn.close()

    def test_hash_tracking_helpers(self, tmp_path, monkeypatch):
        from app.ingestion.watcher import (
            _init_hash_db,
            _is_changed,
            _is_indexed,
            _record_indexed,
        )

        monkeypatch.setattr("app.ingestion.watcher.HASH_DB_PATH", tmp_path / "hashes.db")
        conn = _init_hash_db()
        f = tmp_path / "doc.txt"
        f.write_text("v1", encoding="utf-8")

        assert _is_changed(conn, f) is True   # jamais indexé
        assert _is_indexed(conn, f) is False

        _record_indexed(conn, f, 3)
        assert _is_indexed(conn, f) is True
        assert _is_changed(conn, f) is False

        f.write_text("v2", encoding="utf-8")
        assert _is_changed(conn, f) is True
        assert _is_indexed(conn, f) is False
        conn.close()
