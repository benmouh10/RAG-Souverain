"""
Text-to-SQL engine — moteur de requêtage structuré
====================================================

Chemin alternatif au RAG vectoriel pour les questions structurées/agrégées
(« combien de... », « quelle est la moyenne de... », « liste les 5 plus... »).

Pourquoi un chemin séparé : la recherche par embeddings compare des vecteurs
de *similarité sémantique* entre la question et des fragments de texte. Elle
n'a aucune notion de jointure, de GROUP BY, de comptage exact ou d'agrégat —
elle ne peut donc pas répondre fiablement à « combien d'employés dans le
département Ventes ? » ou « quelle est la moyenne des salaires ? ». Ce module
détecte ce type de question, fait générer une requête SQL par le LLM à partir
du schéma réel de la base SQLite ingérée, exécute cette requête en lecture
seule directement sur le fichier ``.db``, et renvoie le résultat exact — sans
paraphrase ni reformulation des chiffres par le LLM.

Sécurité
--------
Le LLM génère du SQL à partir d'une question utilisateur : il faut donc
traiter ce SQL comme non fiable, au même titre qu'une entrée utilisateur.
Trois couches de défense indépendantes :

1. ``validate_select_only`` — analyse syntaxique légère qui rejette tout ce
   qui n'est pas une unique instruction ``SELECT``/``WITH`` (mots-clés de
   mutation bannis, plusieurs instructions bannies, commentaires bannis).
2. Connexion SQLite ouverte en URI ``mode=ro`` (lecture seule au niveau du
   système de fichiers) + ``PRAGMA query_only = ON`` — même si la couche 1
   laissait passer une requête malveillante, l'exécution échouerait.
3. ``LIMIT`` imposé si absent + garde-fou sur le nombre d'étapes de la
   machine virtuelle SQLite (``set_progress_handler``) pour couper les
   requêtes trop coûteuses (ex. produit cartésien involontaire).
"""

from __future__ import annotations

import html
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("sql_query_engine")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
# Les deux emplacements où l'application écrit réellement les fichiers .db
# uploadés : la page d'administration (data/uploads) et le dossier surveillé
# par le watcher (data/documents). Voir app_ui.ingest_uploaded_file et
# ingestion.watcher.WATCH_DIR.
DB_SEARCH_DIRS = (DATA_DIR / "uploads", DATA_DIR / "documents")

MAX_RESULT_ROWS = 200          # lignes max renvoyées par une requête
MAX_DISPLAY_ROWS = 50          # lignes max affichées dans le tableau HTML
SAMPLE_ROWS_PER_TABLE = 3      # lignes d'exemple données au LLM pour le schéma
MAX_VM_STEPS = 2_000_000       # garde-fou anti requête runaway


# ---------------------------------------------------------------------------
# Détection d'intention : la question est-elle structurée/agrégée ?
# ---------------------------------------------------------------------------
# Bilingue (FR/EN) car le reste de l'app répond dans la langue de la question.
_STRUCTURED_KEYWORDS = [
    # Anglais — aggregation / counting
    r"how many", r"how much", r"count of", r"number of", r"total number",
    r"average", r"\bavg\b", r"\bsum of\b", r"\btotal\b", r"maximum", r"\bmax\b",
    r"minimum", r"\bmin\b", r"median", r"highest", r"lowest", r"most",
    r"least", r"top \d+", r"top-\d+", r"group by", r"sort(ed)? by",
    r"order(ed)? by", r"rank(ed)?", r"percentage", r"percent\b", r"ratio",
    r"distinct", r"unique values", r"list all", r"all records", r"all rows",
    r"compare", r"difference between", r"greater than", r"less than",
    r"more than", r"fewer than", r"older than", r"younger than",
    r"between \d", r"per (table|department|category|month|year)",
    # Français — agrégation / comptage
    r"combien", r"moyenne", r"\bsomme\b", r"\btotal(e|es|aux)?\b", r"maximum",
    r"minimum", r"m[ée]diane", r"le plus\b", r"la plus\b", r"les plus\b",
    r"le moins\b", r"la moins\b", r"les moins\b", r"top \d+", r"trier",
    r"class(er|é|és|ement)", r"nombre de", r"pourcentage", r"taux de",
    r"distinct(e|s|es)?", r"valeurs? uniques?", r"liste(z)? tou(s|tes)",
    r"tou(s|tes) les enregistrements", r"comparer", r"diff[ée]rence entre",
    r"sup[ée]rieur[e]? [àa]", r"inf[ée]rieur[e]? [àa]", r"plus [àa]g[ée]",
    r"plus jeune", r"entre \d", r"par (d[ée]partement|cat[ée]gorie|mois|ann[ée]e)",
]
_STRUCTURED_RE = re.compile("|".join(_STRUCTURED_KEYWORDS), re.IGNORECASE)


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def is_structured_query(question: str) -> bool:
    """Heuristique : la question appelle-t-elle une agrégation/un filtre
    structuré (comptage, moyenne, tri, comparaison, group by...) plutôt
    qu'une recherche sémantique dans du texte libre ?

    Volontairement une heuristique par mots-clés (rapide, déterministe,
    pas d'appel LLM supplémentaire) plutôt qu'un classifieur : le coût d'un
    faux positif est faible (le pipeline SQL retombe sur le RAG en cas
    d'échec de génération/exécution — voir ``RAGEngine.answer``), alors
    qu'un classifieur LLM ajouterait de la latence à chaque message.
    """
    q = _strip_accents(question.strip().lower())
    if not q:
        return False
    return bool(_STRUCTURED_RE.search(q))


# ---------------------------------------------------------------------------
# Découverte des bases SQLite réellement ingérées
# ---------------------------------------------------------------------------
def discover_db_files() -> dict[str, Path]:
    """Retourne ``{nom_de_fichier: chemin}`` pour chaque ``.db`` présent dans
    les répertoires où l'application persiste les fichiers uploadés/watchés.

    C'est la seule source de vérité pour « quelles bases sont interrogeables » :
    un fichier n'apparaît ici que s'il a été réellement uploadé (page Files
    Indexation) ou déposé dans le dossier surveillé — jamais une base devinée
    ou construite par le LLM.
    """
    found: dict[str, Path] = {}
    for directory in DB_SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.db")):
            if path.is_file():
                # Les répertoires sont parcourus dans DB_SEARCH_DIRS (ordre de
                # priorité) : uploads > documents, en cas de nom identique.
                found.setdefault(path.name, path)
    return found


def resolve_db_path(filename: str) -> Path | None:
    """Résout un nom de fichier ``.db`` vers son chemin réel, si connu."""
    return discover_db_files().get(filename)


# ---------------------------------------------------------------------------
# Introspection du schéma
# ---------------------------------------------------------------------------
@dataclass
class TableSchema:
    name: str
    columns: list[tuple[str, str, bool]]  # (nom, type, is_primary_key)
    foreign_keys: list[tuple[str, str, str]]  # (colonne, table_référencée, colonne_référencée)
    row_count: int
    sample_rows: list[tuple] = field(default_factory=list)


def _connect_readonly(db_path: Path, timeout: float = 10.0) -> sqlite3.Connection:
    """Ouvre une connexion strictement en lecture seule.

    ``mode=ro`` empêche toute écriture au niveau du VFS SQLite lui-même :
    même une requête qui aurait franchi ``validate_select_only`` par erreur
    échouerait ici avec ``sqlite3.OperationalError: attempt to write a
    readonly database``.
    """
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=timeout)
    conn.execute("PRAGMA query_only = ON;")
    return conn


def get_tables(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    return [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]


def inspect_schema(db_path: Path) -> list[TableSchema]:
    """Introspecte toutes les tables d'une base SQLite : colonnes, types,
    clés primaires/étrangères, nombre de lignes et quelques lignes
    d'exemple — tout ce dont le LLM a besoin pour générer un SQL correct
    sans jamais halluciner un nom de table/colonne.
    """
    conn = _connect_readonly(db_path)
    try:
        tables: list[TableSchema] = []
        for table_name in get_tables(conn):
            # Nom de table validé contre sqlite_master juste au-dessus : pas
            # d'injection possible via un f-string ici (identifiants, pas des
            # valeurs — sqlite3 ne paramètre pas les identifiants).
            col_rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            columns = [(row[1], row[2] or "TEXT", bool(row[5])) for row in col_rows]

            fk_rows = conn.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
            foreign_keys = [(row[3], row[2], row[4]) for row in fk_rows]

            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM '{table_name}'").fetchone()[0]
            except sqlite3.Error:
                row_count = -1

            sample_rows: list[tuple] = []
            try:
                sample_rows = conn.execute(
                    f"SELECT * FROM '{table_name}' LIMIT {SAMPLE_ROWS_PER_TABLE}"
                ).fetchall()
            except sqlite3.Error:
                pass

            tables.append(
                TableSchema(
                    name=table_name,
                    columns=columns,
                    foreign_keys=foreign_keys,
                    row_count=row_count,
                    sample_rows=sample_rows,
                )
            )
        return tables
    finally:
        conn.close()


def format_schema_for_prompt(tables: list[TableSchema]) -> str:
    """Sérialise le schéma en texte compact pour le prompt de génération SQL."""
    if not tables:
        return "(no tables found)"

    blocks = []
    for table in tables:
        col_lines = []
        for name, col_type, is_pk in table.columns:
            marker = " PRIMARY KEY" if is_pk else ""
            col_lines.append(f"  - {name} ({col_type}{marker})")
        for col, ref_table, ref_col in table.foreign_keys:
            col_lines.append(f"    -> {col} references {ref_table}.{ref_col}")

        block = [f"Table: {table.name} ({table.row_count} rows)", "Columns:"] + col_lines
        if table.sample_rows:
            col_names = [c[0] for c in table.columns]
            block.append(f"Sample rows ({', '.join(col_names)}):")
            for row in table.sample_rows:
                block.append(f"  {tuple(row)}")
        blocks.append("\n".join(block))

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Validation de sécurité du SQL généré par le LLM
# ---------------------------------------------------------------------------
class SQLSafetyError(ValueError):
    """Le SQL généré par le LLM ne respecte pas la politique lecture seule."""


class SQLGenerationError(RuntimeError):
    """Le LLM n'a pas pu générer de requête pertinente pour la question."""


_BANNED_KEYWORDS_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|"
    r"ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|TRIGGER|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT|"
    r"EXEC|EXECUTE|GRANT|REVOKE"
    r")\b",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def validate_select_only(raw_sql: str) -> str:
    """Nettoie et valide le SQL renvoyé par le LLM.

    Lève ``SQLSafetyError`` si la requête n'est pas une unique instruction
    ``SELECT``/``WITH`` en lecture seule. Ajoute un ``LIMIT`` si absent.
    """
    cleaned = _FENCE_RE.sub("", raw_sql or "").strip()
    if not cleaned:
        raise SQLSafetyError("Empty SQL query generated.")

    # Une seule instruction : les points-virgules internes (hors chaîne)
    # sont rares dans des requêtes analytiques ; on refuse tout ce qui
    # ressemble à plusieurs instructions empilées.
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    if len(statements) != 1:
        raise SQLSafetyError("Only a single SQL statement is allowed.")
    cleaned = statements[0]

    if not re.match(r"^(SELECT|WITH)\b", cleaned, re.IGNORECASE):
        raise SQLSafetyError("Only SELECT queries are allowed.")

    if _BANNED_KEYWORDS_RE.search(cleaned):
        raise SQLSafetyError("Query contains a disallowed keyword.")

    if "--" in cleaned or "/*" in cleaned:
        raise SQLSafetyError("SQL comments are not allowed in generated queries.")

    if not re.search(r"\bLIMIT\b", cleaned, re.IGNORECASE):
        cleaned = f"{cleaned} LIMIT {MAX_RESULT_ROWS}"

    return cleaned


# ---------------------------------------------------------------------------
# Exécution
# ---------------------------------------------------------------------------
def execute_readonly_sql(db_path: Path, sql: str) -> tuple[list[str], list[tuple]]:
    """Exécute une requête déjà validée contre la base SQLite, en lecture
    seule. Retourne ``(colonnes, lignes)``.
    """
    conn = _connect_readonly(db_path)
    try:
        steps = {"n": 0}

        def _progress_handler() -> int:
            steps["n"] += 1
            return 1 if steps["n"] > MAX_VM_STEPS else 0

        conn.set_progress_handler(_progress_handler, 1000)

        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, rows
    except sqlite3.ProgrammingError as exc:
        raise SQLSafetyError(f"SQLite refused the query: {exc}") from exc
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rendu du résultat (HTML — cohérent avec le style du reste de l'app,
# voir app.css et pages.chat._render_source_drawer)
# ---------------------------------------------------------------------------
def render_result_html(columns: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "<div class='sql-answer'>The query ran successfully but returned no matching rows.</div>"

    if len(rows) == 1 and len(columns) == 1:
        value = rows[0][0]
        value_str = "0" if value is None else str(value)
        return f"<div class='sql-answer'><strong>Answer:</strong> {html.escape(value_str)}</div>"

    display_rows = rows[:MAX_DISPLAY_ROWS]
    thead = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = []
    for row in display_rows:
        cells = "".join(
            f"<td>{html.escape('—' if v is None else str(v))}</td>" for v in row
        )
        body.append(f"<tr>{cells}</tr>")
    table_html = (
        "<table class='sql-result-table file-table'>"
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )
    note = ""
    if len(rows) > len(display_rows):
        note = f"<div class='sql-note'>Showing {len(display_rows)} of {len(rows)} row(s).</div>"

    header = f"<div class='sql-answer'><strong>{len(rows)} row(s) found.</strong></div>"
    return header + table_html + note


_ROUTING_DB_RE = re.compile(r"DATABASE:\s*(\S+)", re.IGNORECASE)
_ROUTING_SQL_RE = re.compile(r"SQL:\s*(.*)", re.IGNORECASE | re.DOTALL)


def parse_routed_response(text: str) -> tuple[str | None, str]:
    """Parse the ``DATABASE: <name>\\nSQL: <query>`` response produced by
    ``OllamaLLM.generate_routed_sql`` for the multi-database case.

    Returns ``(None, "")`` if the model declined (``NO_QUERY``) or the
    response doesn't match the expected shape — callers must treat both as
    "could not route" rather than guessing.
    """
    if not text or text.strip().upper().startswith("NO_QUERY"):
        return None, ""
    db_match = _ROUTING_DB_RE.search(text)
    sql_match = _ROUTING_SQL_RE.search(text)
    if not db_match or not sql_match:
        return None, ""
    return db_match.group(1).strip(), sql_match.group(1).strip()


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------
@dataclass
class SQLAnswer:
    answer_html: str
    sql: str
    db_file: str
    columns: list[str]
    rows: list[tuple]


class SQLQueryEngine:
    """Point d'entrée unique du chemin text-to-SQL : sélection de la base,
    génération de la requête par le LLM, validation, exécution, rendu.

    Résolution de la base à interroger, par ordre de priorité :
    1. ``file_hint`` explicite (ex. le sélecteur « Target Document » du chat).
    2. Un nom de fichier littéralement présent dans la question (« @sales.db
       combien de... » fonctionne déjà, la recherche est une simple
       sous-chaîne — pas besoin d'une syntaxe ``@`` spéciale).
    3. Une seule base ingérée : aucune ambiguïté possible.
    4. Plusieurs bases, aucun indice : le LLM reçoit le schéma de *toutes*
       les bases et doit lui-même désigner celle qui contient les données
       pertinentes avant de générer la requête (``generate_routed_sql``) —
       remplace l'ancienne heuristique « base la plus récemment modifiée »,
       qui n'avait aucune idée du contenu réel des bases.
    """

    def __init__(self) -> None:
        self._schema_cache: dict[str, tuple[float, str]] = {}  # name -> (mtime, schema_text)

    # -- Découverte -----------------------------------------------------
    def has_databases(self) -> bool:
        return bool(discover_db_files())

    def list_databases(self) -> list[str]:
        return sorted(discover_db_files().keys())

    def _match_mentioned_db(
        self, question: str, databases: dict[str, Path]
    ) -> tuple[str, Path] | None:
        q_lower = question.lower()
        for name, path in databases.items():
            if name.lower() in q_lower:
                return name, path
        return None

    def _finalize_sql(self, raw_sql: str) -> str:
        """Shared NO_QUERY check + safety validation, used by every
        resolution path so the guarantees are identical regardless of how
        the target database was chosen."""
        if not raw_sql or raw_sql.strip().upper().startswith("NO_QUERY"):
            raise SQLGenerationError(
                "The LLM could not map this question to the available schema."
            )
        return validate_select_only(raw_sql)

    # -- Schéma (avec cache invalidé par mtime) --------------------------
    def _get_schema_text(self, name: str, path: Path) -> str:
        mtime = path.stat().st_mtime
        cached = self._schema_cache.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        tables = inspect_schema(path)
        schema_text = format_schema_for_prompt(tables)
        self._schema_cache[name] = (mtime, schema_text)
        return schema_text

    # -- Pipeline complet -------------------------------------------------
    def answer(self, question: str, llm: Any, file_hint: str | None = None) -> SQLAnswer:
        """Exécute le pipeline text-to-SQL complet pour ``question``.

        Lève ``SQLGenerationError``/``SQLSafetyError``/``sqlite3.Error`` en
        cas d'échec à n'importe quelle étape — à charge de l'appelant de
        retomber sur le pipeline RAG classique (voir ``RAGEngine.answer``).
        """
        databases = discover_db_files()
        if not databases:
            raise SQLGenerationError("No SQLite database has been ingested yet.")

        resolved = None
        if file_hint and file_hint in databases:
            resolved = (file_hint, databases[file_hint])
        else:
            resolved = self._match_mentioned_db(question, databases)
            if resolved is None and len(databases) == 1:
                resolved = next(iter(databases.items()))

        if resolved is not None:
            db_name, db_path = resolved
            schema_text = self._get_schema_text(db_name, db_path)
            raw_sql = llm.generate_sql(question, schema_text)
            sql = self._finalize_sql(raw_sql)
        else:
            # Plusieurs bases, aucun indice explicite : on laisse le LLM
            # choisir en lui montrant tous les schémas d'un coup, plutôt que
            # de deviner (voir docstring de la classe).
            schemas = {name: self._get_schema_text(name, path) for name, path in databases.items()}
            routed_response = llm.generate_routed_sql(question, schemas)
            db_name, raw_sql = parse_routed_response(routed_response)
            if db_name is None:
                raise SQLGenerationError(
                    "The LLM could not map this question to any of the available databases."
                )
            if db_name not in databases:
                raise SQLGenerationError(
                    f"The LLM selected an unknown database ({db_name!r}); refusing to guess."
                )
            db_path = databases[db_name]
            sql = self._finalize_sql(raw_sql)

        columns, rows = execute_readonly_sql(db_path, sql)
        answer_html = render_result_html(columns, rows)

        logger.info(
            "Text-to-SQL: db=%s rows=%d sql=%s", db_name, len(rows), sql.replace("\n", " ")
        )
        return SQLAnswer(
            answer_html=answer_html,
            sql=sql,
            db_file=db_name,
            columns=columns,
            rows=rows,
        )

