"""Centralized SQLite database management for RAG Souverain.

Tables:
    users         — account and role management
    doc_versions  — indexed document version history
    conversations — multi-session chat per user
    chat_history  — conversation history by user_id + conversation_id
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("rag_database")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("RAG_DB_PATH", str(PROJECT_ROOT / "data" / "rag.db")))


def _hash_password(password: str, salt: str = "") -> str:
    if not salt:
        salt = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    hashed = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return f"{salt}${hashed}"


def _verify_password(stored: str, password: str) -> bool:
    if "$" not in stored:
        return False
    salt, _ = stored.split("$", 1)
    return _hash_password(password, salt) == stored


def _is_none_string(val: Any) -> bool:
    """Check if value is a 'None' string variant (case-insensitive, whitespace-tolerant)."""
    if not isinstance(val, str):
        return False
    return val.strip().lower() == "none"


def _sanitize_val(val: Any, default: Any = "") -> Any:
    """Return val if it's a meaningful value, otherwise default."""
    if val is None:
        return default
    if _is_none_string(val):
        return default
    if isinstance(val, str) and not val.strip():
        return default
    return val


def _sanitize_sources(sources: Any) -> list[dict[str, Any]]:
    """Ensure every source dict has safe default values (no None leaks)."""
    if not isinstance(sources, list):
        return []
    sanitized = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        sanitized.append({
            "content": _sanitize_val(s.get("content")),
            "score": _sanitize_val(s.get("score"), 0.0),
            "source_file": _sanitize_val(s.get("source_file"), "Unknown source"),
            "page_number": _sanitize_val(s.get("page_number"), -1),
            "chunk_index": _sanitize_val(s.get("chunk_index"), -1),
            "version": _sanitize_val(s.get("version"), "-"),
        })
    return sanitized


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class User:
    """Compte utilisateur de l'application.

    ``role`` est l'un des rôles RBAC définis dans ``session.py``
    ("admin" ou "user"). ``created_at`` est un timestamp UNIX.
    """

    user_id: int
    username: str
    role: str
    created_at: float


@dataclass
class DocVersion:
    """Version d'un document indexé, pour le versionnage des fichiers.

    Le versionnage est **global** (par ``filename``), partagé entre tous les
    administrateurs : un même fichier ré-indexé incrémente ``version`` ;
    ``chunk_count`` permet de connaître le nombre de chunks associés.
    """

    doc_id: int
    filename: str
    file_hash: str
    version: int
    chunk_count: int
    indexed_at: float


class Database:
    """Gestion centralisée SQLite — users + versions + chat_history."""

    def __init__(self, db_path: str | Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                    username   TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role       TEXT NOT NULL DEFAULT 'user',
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS doc_versions (
                    doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename    TEXT NOT NULL,
                    file_hash   TEXT NOT NULL,
                    version     INTEGER NOT NULL DEFAULT 1,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    indexed_at  REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conv_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    title       TEXT NOT NULL DEFAULT 'Nouvelle conversation',
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS chat_history (
                    chat_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id        INTEGER NOT NULL,
                    conversation_id INTEGER NOT NULL DEFAULT 0,
                    role           TEXT NOT NULL,
                    content        TEXT NOT NULL,
                    sources        TEXT DEFAULT '[]',
                    created_at     REAL NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                );
            """)
            conn.commit()

            # Migration: add conversation_id to existing chat_history if missing
            cursor = conn.execute("PRAGMA table_info(chat_history)")
            columns = {row[1] for row in cursor.fetchall()}
            if "conversation_id" not in columns:
                conn.execute("ALTER TABLE chat_history ADD COLUMN conversation_id INTEGER NOT NULL DEFAULT 0")
                conn.commit()

            # Migration: add conversations table rows for existing users with chat history
            existing = conn.execute("SELECT COUNT(*) as cnt FROM conversations").fetchone()
            if existing["cnt"] == 0:
                users_with_chats = conn.execute(
                    "SELECT DISTINCT user_id FROM chat_history WHERE conversation_id = 0"
                ).fetchall()
                for row in users_with_chats:
                    now = time.time()
                    conn.execute(
                        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                        (row["user_id"], "Conversation", now, now),
                    )
                    conv_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    conn.execute(
                        "UPDATE chat_history SET conversation_id = ? WHERE user_id = ? AND conversation_id = 0",
                        (conv_id, row["user_id"]),
                    )
                conn.commit()

            # Migration C1/Mo4: global document versioning (drop per-user key).
            # Rebuild doc_versions without the user_id column (global by filename).
            doc_versions_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(doc_versions)").fetchall()
            }
            if "user_id" in doc_versions_cols:
                conn.executescript(
                    """
                    CREATE TABLE doc_versions_new (
                        doc_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename    TEXT NOT NULL,
                        file_hash   TEXT NOT NULL,
                        version     INTEGER NOT NULL DEFAULT 1,
                        chunk_count INTEGER NOT NULL DEFAULT 0,
                        indexed_at  REAL NOT NULL
                    );
                    INSERT INTO doc_versions_new (doc_id, filename, file_hash, version, chunk_count, indexed_at)
                        SELECT doc_id, filename, file_hash, version, chunk_count, indexed_at FROM doc_versions;
                    DROP TABLE doc_versions;
                    ALTER TABLE doc_versions_new RENAME TO doc_versions;
                    """
                )
                conn.commit()

        if self.get_user_by_username("admin") is None:
            self.create_user("admin", "admin", role="admin")
            logger.info("Default admin user created (password: admin)")

    # ── Users ──────────────────────────────────────────────────────────────

    def create_user(self, username: str, password: str, role: str = "user") -> User | None:
        """Crée un utilisateur ; retourne ``None`` si le nom est déjà pris.

        Le mot de passe n'est jamais stocké en clair : il est haché via
        ``_hash_password`` (scrypt avec sel aléatoire) avant insertion.
        """
        if self.get_user_by_username(username) is not None:
            return None
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                (username, _hash_password(password), role, now),
            )
            conn.commit()
            return User(user_id=cursor.lastrowid, username=username, role=role, created_at=now)

    def authenticate(self, username: str, password: str) -> User | None:
        """Vérifie un couple identifiant/mot de passe.

        La comparaison se fait sur le hachage via ``_verify_password``
        (temps constant) ; on ne retourne l'utilisateur qu'en cas de
        correspondance, pour éviter de fuiter l'existence du compte.
        """
        user = self.get_user_by_username(username)
        if user is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE user_id = ?", (user.user_id,)
            ).fetchone()
        if row is None:
            return None
        return user if _verify_password(row["password_hash"], password) else None

    def get_user_by_username(self, username: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            return None
        return User(user_id=row["user_id"], username=row["username"], role=row["role"], created_at=row["created_at"])

    def get_user_by_id(self, user_id: int) -> User | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return User(user_id=row["user_id"], username=row["username"], role=row["role"], created_at=row["created_at"])

    def list_users(self) -> list[User]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        return [User(user_id=r["user_id"], username=r["username"], role=r["role"], created_at=r["created_at"]) for r in rows]

    def change_password(self, user_id: int, new_password: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (_hash_password(new_password), user_id))
            conn.commit()
            return cursor.rowcount > 0

    def delete_user(self, user_id: int) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ── Document versioning ────────────────────────────────────────────────

    def record_ingestion(self, filename: str, file_hash: str, chunk_count: int) -> DocVersion:
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT version FROM doc_versions WHERE filename = ? ORDER BY version DESC LIMIT 1",
                (filename,),
            ).fetchone()

            new_version = (existing["version"] + 1) if existing else 1
            now = time.time()
            cursor = conn.execute(
                "INSERT INTO doc_versions (filename, file_hash, version, chunk_count, indexed_at) VALUES (?, ?, ?, ?, ?)",
                (filename, file_hash, new_version, chunk_count, now),
            )
            conn.commit()
            return DocVersion(
                doc_id=cursor.lastrowid,
                filename=filename,
                file_hash=file_hash,
                version=new_version,
                chunk_count=chunk_count,
                indexed_at=now,
            )

    def get_latest_version(self, filename: str) -> DocVersion | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM doc_versions WHERE filename = ? ORDER BY version DESC LIMIT 1",
                (filename,),
            ).fetchone()
        if row is None:
            return None
        return DocVersion(
            doc_id=row["doc_id"], filename=row["filename"],
            file_hash=row["file_hash"], version=row["version"],
            chunk_count=row["chunk_count"], indexed_at=row["indexed_at"],
        )

    def get_file_hash(self, filename: str) -> str | None:
        v = self.get_latest_version(filename)
        return v.file_hash if v else None

    def list_documents(self) -> list[DocVersion]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM doc_versions ORDER BY indexed_at DESC",
            ).fetchall()
        return [
            DocVersion(
                doc_id=r["doc_id"], filename=r["filename"],
                file_hash=r["file_hash"], version=r["version"],
                chunk_count=r["chunk_count"], indexed_at=r["indexed_at"],
            )
            for r in rows
        ]

    def get_total_chunks(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) as total FROM doc_versions",
            ).fetchone()
        return row["total"] if row else 0

    # ── Chat history persistence ───────────────────────────────────────────

    def create_conversation(self, user_id: int, title: str = "Nouvelle conversation") -> int:
        """Crée une nouvelle conversation et retourne son ID."""
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (user_id, title, now, now),
            )
            conn.commit()
            return cursor.lastrowid

    def list_conversations(self, user_id: int) -> list[dict[str, Any]]:
        """Liste les conversations d'un utilisateur, plus récentes en premier."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT conv_id, title, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, conv_id: int, title: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conv_id = ?",
                (title, time.time(), conv_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_conversation(self, conv_id: int) -> bool:
        """Supprime une conversation et tous ses messages."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_history WHERE conversation_id = ?", (conv_id,))
            cursor = conn.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))
            conn.commit()
            return cursor.rowcount > 0

    def save_chat_message(
        self, user_id: int, role: str, content: str,
        sources: list[dict[str, Any]] | None = None,
        conversation_id: int = 0,
    ) -> None:
        import json
        now = time.time()
        safe_content = content if content and content.strip() and content.strip() != "None" else ""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO chat_history (user_id, conversation_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, conversation_id, role, safe_content, json.dumps(sources or [], ensure_ascii=False), now),
            )
            # Update conversation timestamp
            if conversation_id:
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conv_id = ?",
                    (now, conversation_id),
                )
            conn.commit()

    def get_chat_history(self, user_id: int, conversation_id: int | None = None, limit: int = 200) -> list[dict[str, Any]]:
        import json
        with self._connect() as conn:
            if conversation_id is not None:
                rows = conn.execute(
                    "SELECT role, content, sources, created_at FROM chat_history WHERE user_id = ? AND conversation_id = ? ORDER BY created_at ASC LIMIT ?",
                    (user_id, conversation_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, sources, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at ASC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        return [
            {
                "role": r["role"],
                "content": "" if _is_none_string(r["content"]) or not (r["content"] and r["content"].strip()) else r["content"],
                "sources": _sanitize_sources(json.loads(r["sources"])) if r["sources"] else [],
            }
            for r in rows
        ]

    def clear_chat_history(self, user_id: int, conversation_id: int | None = None) -> None:
        with self._connect() as conn:
            if conversation_id is not None:
                conn.execute("DELETE FROM chat_history WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id))
            else:
                conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
            conn.commit()

    def cleanup_old_messages(self, max_age_days: int = 30) -> int:
        """Supprime les messages plus anciens que max_age_days. Retourne le nombre supprimé."""
        cutoff = time.time() - (max_age_days * 86400)
        with self._connect() as conn:
            # Delete orphaned conversations (no messages left)
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE created_at < ?", (cutoff,)
            )
            deleted = cursor.rowcount
            # Clean up conversations with no messages
            conn.execute(
                """DELETE FROM conversations WHERE conv_id NOT IN
                   (SELECT DISTINCT conversation_id FROM chat_history WHERE conversation_id != 0)"""
            )
            conn.commit()
        return deleted

    def cleanup_none_messages(self, user_id: int) -> int:
        """Supprime les messages dont le contenu est 'None' (toutes variantes)."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_history WHERE user_id = ? AND (content IS NULL OR content = '' OR LOWER(TRIM(content)) = 'none')",
                (user_id,),
            )
            deleted = cursor.rowcount
            conn.commit()
        return deleted
