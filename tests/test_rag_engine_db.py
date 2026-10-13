"""Tests for the active-DB path handling in app/core/rag_engine.py."""

from __future__ import annotations

from pathlib import Path

from app.core.rag_engine import _ensure_active_db_file


class TestEnsureActiveDbFile:
    def test_noop_when_missing(self, tmp_path):
        path = tmp_path / "active_rag.db"
        _ensure_active_db_file(path)
        assert not path.exists()

    def test_noop_when_real_file(self, tmp_path):
        path = tmp_path / "active_rag.db"
        path.write_text("stale")
        _ensure_active_db_file(path)
        assert path.is_file()
        assert path.read_text() == "stale"

    def test_moves_stray_directory_aside(self, tmp_path):
        path = tmp_path / "active_rag.db"
        path.mkdir()
        (path / "important.db").write_text("keep me")

        _ensure_active_db_file(path)

        assert not path.exists()
        backup = tmp_path / "active_rag.db.unused"
        assert backup.is_dir()
        assert (backup / "important.db").read_text() == "keep me"

    def test_picks_free_backup_name(self, tmp_path):
        path = tmp_path / "active_rag.db"
        path.mkdir()
        (tmp_path / "active_rag.db.unused").write_text("taken")

        _ensure_active_db_file(path)

        assert (tmp_path / "active_rag.db.unused").read_text() == "taken"
        assert (tmp_path / "active_rag.db.unused.1").is_dir()