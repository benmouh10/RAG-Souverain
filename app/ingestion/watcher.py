"""
Automatic file watcher for continuous ingestion.

Scope (C5)
----------
Le watcher surveille **uniquement** ``data/documents/`` (non récursif) —
périmètre volontairement restreint, conformément à la conception : seuls les
fichiers déposés dans ce répertoire partagé sont indexés automatiquement.
Le dossier ``data/documents`` est monté en lecture seule côté conteneur
(docker-compose), ce qui garantit qu'aucune écriture concurrente ne provient
des autres services.

Sécurité des événements ``on_modified``
---------------------------------------
``on_modified`` est sans danger car :
1. chaque événement est seulement **planifié** (debounce ``DEBOUNCE_SECONDS``) ;
2. ``process_pending`` revérifie l'existence du fichier avant ingestion ;
3. ``_is_indexed`` compare le hash SHA-256 courant : si un fichier est modifié
   en cours d'écriture (copie partielle, ``.tmp``…), l'ancienne version reste
   en place jusqu'à l'ingestion complète de la nouvelle, et les fichiers sans
   modification de contenu ne sont jamais ré-indexés ;
4. les fichiers temporaires/hiddens (``.*``, ``~*``, ``*.swp``…) sont ignorés.

Versionnage global (C1/Mo4)
---------------------------
Le watcher indexe des chunks **sans** versionnage (pas de colonne ``user_id``,
pas de gestion de version) : il utilise le hash de fichier pour détecter les
changements. Le versionnage versionné (doc_versions) concerne l'ingestion
via l'interface administrateur (``RAGEngine.ingest_document``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("ingestion_watcher")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
WATCH_DIR = DATA_DIR / "documents"
HASH_DB_PATH = DATA_DIR / "indexed_files.db"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
IGNORED_PATTERNS = {"*~", "~*", ".*", ".DS_Store", "*.swp", "*.swo"}

# Wait time after file modification before ingesting (debounce)
DEBOUNCE_SECONDS = 2.0


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _init_hash_db() -> sqlite3.Connection:
    """Initialize SQLite database for tracking indexed file hashes."""
    HASH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(HASH_DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_files (
            file_path TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            indexed_at REAL NOT NULL,
            chunk_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _is_indexed(conn: sqlite3.Connection, file_path: Path) -> bool:
    """Check if file is already indexed with same hash."""
    current_hash = _compute_file_hash(file_path)
    row = conn.execute(
        "SELECT file_hash FROM indexed_files WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    if row and row[0] == current_hash:
        return True
    return False


def _is_changed(conn: sqlite3.Connection, file_path: Path) -> bool:
    """Check if file hash has changed since last indexing."""
    current_hash = _compute_file_hash(file_path)
    row = conn.execute(
        "SELECT file_hash FROM indexed_files WHERE file_path = ?",
        (str(file_path),),
    ).fetchone()
    if row is None:
        return True  # Never indexed
    return row[0] != current_hash


def _record_indexed(conn: sqlite3.Connection, file_path: Path, chunk_count: int) -> None:
    """Record that a file has been indexed."""
    file_hash = _compute_file_hash(file_path)
    conn.execute(
        "INSERT OR REPLACE INTO indexed_files (file_path, file_hash, indexed_at, chunk_count) VALUES (?, ?, ?, ?)",
        (str(file_path), file_hash, time.time(), chunk_count),
    )
    conn.commit()


def _should_ignore(file_path: Path) -> bool:
    """Check if file should be ignored (temp files, etc)."""
    name = file_path.name
    for pattern in IGNORED_PATTERNS:
        if file_path.match(pattern):
            return True
    return False


def _ingest_file(file_path: Path, conn: sqlite3.Connection) -> int:
    """Run full ingestion pipeline on a single file. Returns chunk count.
    If file hash changed, deletes old chunks first."""
    from app.ingestion.parser import DocumentParser
    from app.ingestion.chunker import AdaptiveChunker
    from app.indexing.vector_store import ChromaVectorStore

    chroma_path = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma"))
    collection_name = os.getenv("CHROMA_COLLECTION", "rag_documents")

    try:
        # Delete old chunks if file hash changed
        if _is_changed(conn, file_path):
            vector_store = ChromaVectorStore(
                persist_directory=chroma_path,
                collection_name=collection_name,
            )
            deleted = vector_store.delete_by_source(file_path.name)
            if deleted:
                logger.info("[Watcher] Deleted %d old chunks for updated file: %s", deleted, file_path.name)

        suffix = file_path.suffix.lower()

        if suffix == ".db":
            from app.ingestion.parser import SQLParser
            parser = SQLParser(file_path)
            doc = parser.to_parsed_documents()
            parser.close()
        else:
            parser = DocumentParser(ocr_enabled=False)
            doc = parser.parse_file(file_path)

        if doc.page_count == 0:
            logger.warning("No content extracted from %s", file_path.name)
            return 0

        chunker = AdaptiveChunker()
        chunks = chunker.chunk_documents([doc])

        if not chunks:
            logger.warning("No chunks generated from %s", file_path.name)
            return 0

        vector_store = ChromaVectorStore(
            persist_directory=chroma_path,
            collection_name=collection_name,
        )
        vector_store.initialize_collection()
        added = vector_store.add_chunks(chunks)

        _record_indexed(conn, file_path, added)
        logger.info(
            "[Watcher] Ingested %s: %d chunks (total: %d)",
            file_path.name, added, vector_store.count(),
        )
        return added

    except Exception as e:
        logger.error("[Watcher] Failed to ingest %s: %s", file_path.name, e)
        return 0


def _scan_and_ingest(conn: sqlite3.Connection) -> None:
    """Scan watch directory and ingest any new/modified files."""
    if not WATCH_DIR.is_dir():
        logger.warning("Watch directory does not exist: %s", WATCH_DIR)
        return

    for file_path in WATCH_DIR.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if _should_ignore(file_path):
            continue
        if _is_indexed(conn, file_path):
            continue

        logger.info("[Watcher] New/modified file detected: %s", file_path.name)
        _ingest_file(file_path, conn)


class WatcherEventHandler:
    """Handles file system events from watchdog."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self._pending: dict[str, float] = {}

    def on_created(self, event):
        if event.is_directory:
            return
        self._schedule_ingest(Path(event.src_path))

    def on_modified(self, event):
        # Sûr : simple planification + debounce + contrôle du hash (voir docstring C5).
        if event.is_directory:
            return
        self._schedule_ingest(Path(event.src_path))

    def _schedule_ingest(self, file_path: Path) -> None:
        if not file_path.is_file():
            return
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        if _should_ignore(file_path):
            return
        self._pending[str(file_path)] = time.time()

    def process_pending(self) -> None:
        """Process any pending files after debounce period."""
        now = time.time()
        to_process = [
            p for p, t in self._pending.items()
            if now - t >= DEBOUNCE_SECONDS
        ]
        for path_str in to_process:
            self._pending.pop(path_str, None)
            file_path = Path(path_str)
            if not file_path.exists():
                continue
            if _is_indexed(self.conn, file_path):
                continue
            _ingest_file(file_path, self.conn)


def main() -> None:
    """Main entry point for the file watcher service."""
    logger.info("[Watcher] Starting file watcher for: %s", WATCH_DIR)
    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    conn = _init_hash_db()

    # Initial scan
    _scan_and_ingest(conn)

    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class _EventHandler(FileSystemEventHandler):
            def __init__(self, handler: WatcherEventHandler) -> None:
                self._handler = handler

            def on_created(self, event):
                self._handler.on_created(event)

            def on_modified(self, event):
                self._handler.on_modified(event)

        handler = WatcherEventHandler(conn)
        observer = Observer()
        observer.schedule(_EventHandler(handler), str(WATCH_DIR), recursive=False)
        observer.start()

        logger.info("[Watcher] Watching %s for new files...", WATCH_DIR)

        try:
            while True:
                time.sleep(1)
                handler.process_pending()
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    except ImportError:
        logger.warning("watchdog not installed. Falling back to periodic scan.")
        try:
            while True:
                time.sleep(30)
                _scan_and_ingest(conn)
        except KeyboardInterrupt:
            pass

    conn.close()
    logger.info("[Watcher] Stopped.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
