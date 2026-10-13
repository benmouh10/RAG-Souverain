"""
Module de generation (LLM) pour le systeme RAG.

Gere la communication avec le LLM local (Ollama) et la construction du prompt.
Implemente le streaming de la reponse, les instructions anti-hallucination,
et la memoire de conversation multi-tours.
"""

import json
import logging
import os
import time
import requests
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = "mistral"
DEFAULT_TEMPERATURE = 0.1
TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT", "300"))
CONVERSATION_WINDOW = int(os.getenv("CONVERSATION_WINDOW", "3"))
USE_STRUCTURED_OUTPUT = os.getenv("USE_STRUCTURED_OUTPUT", "false").lower() in ("true", "1", "yes")

# M2 — Health check & circuit breaker
HEALTH_CHECK_TIMEOUT = 10.0
CIRCUIT_BREAKER_FAILURES = 3      # consecutive failures before opening
CIRCUIT_BREAKER_COOLDOWN = 30.0   # duration of opening (seconds)

SYSTEM_PROMPT = """You are a precise, rigorous, and autonomous corporate assistant. Answer user questions based solely on the provided document extracts and database records.

IMPORTANT INSTRUCTIONS:
1. GROUNDING: Base your answer ONLY on the CONTEXT extracts below. If the exact information is present (even partially), provide a complete answer by combining what is available. NEVER invent facts, figures, names, or definitions that are not in the CONTEXT.
2. DEFINITIONS: When the user asks "what is X?" and the CONTEXT defines X explicitly, reuse that definition faithfully. Do NOT replace it with a near-synonym concept from another extract (e.g. do not present the "RAGAS evaluation framework" as the definition of the "RAG system").
3. DISTINGUISH CONCEPTS: If several related-but-different notions appear (system, framework, module, product), keep them distinct and describe each with its own extract. Do not merge them into one invented name.
4. CONVERSATION HISTORY IS NOT A SOURCE: The HISTORIQUE DE CONVERSATION block is only conversational context (what was said before). It is NOT part of the document extracts. Never import facts, quotes, file names, or definitions from the history into your answer — ground everything in CONTEXT only.
5. ASSOCIATIONS & MATCHES: If a question concerns a patient, employee, or project (e.g., Omar, Sara, Atlas), gather all related data rows or paragraphs to provide a complete answer.
6. ABSENCE OF INFORMATION: If the requested information is completely absent from the extracts, respond clearly and briefly: "The information is not available in the provided documents." Do not guess.
7. CITATION: Mention the sources used (file name .pdf, .csv, or .db / table) ONLY if they genuinely appear in the CONTEXT. Never cite an extract, file, or page that is not present.

LANGUAGE: Respond in the same language as the user's question.

CONTEXT:
{context}"""

SQL_GENERATION_SYSTEM_PROMPT = """You are an expert SQLite query writer. Given a database schema and a natural-language question, write ONE single valid, read-only SQLite query that answers the question exactly.

STRICT RULES:
1. Output ONLY the raw SQL query. No explanation, no markdown code fences, no comments, no trailing semicolon.
2. Use ONLY the tables and columns listed in the schema below, with their exact names. Never invent a table or column.
3. The query MUST start with SELECT or WITH. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA, or any statement other than a single read query.
4. Use JOINs (based on the foreign keys shown in the schema) when the question spans multiple tables.
5. Use aggregate functions (COUNT, SUM, AVG, MIN, MAX), GROUP BY, ORDER BY and LIMIT as needed to answer precisely and completely.
6. Prefer explicit column names over SELECT * when the question asks for specific fields.
7. If the question genuinely cannot be answered from the schema below (missing table/column), output exactly: NO_QUERY

DATABASE SCHEMA:
{schema}"""

SQL_ROUTING_SYSTEM_PROMPT = """You are an expert SQLite query writer with access to MULTIPLE separate databases. Given the schemas of every available database and a natural-language question, decide which ONE database actually contains the data needed, then write ONE single valid, read-only SQLite query against that database only.

STRICT RULES:
1. Pick exactly one database — the one whose tables/columns genuinely match the question. Never combine tables from two different databases in one query (they are separate files and cannot be joined directly).
2. Respond in EXACTLY this two-line format and nothing else (no explanation, no markdown fences):
DATABASE: <database file name, copied exactly from "--- Database: ... ---" below>
SQL: <the SQL query, no comments, no trailing semicolon>
3. The query MUST start with SELECT or WITH. Never write INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH, PRAGMA, or any statement other than a single read query.
4. Use ONLY tables/columns that literally appear in the chosen database's schema. Never invent a table, column, or database name.
5. Use aggregate functions, JOINs (within the chosen database only), GROUP BY, ORDER BY and LIMIT as needed to answer precisely.
6. If NONE of the databases below can answer the question, respond with exactly: NO_QUERY

AVAILABLE DATABASES:
{schemas}"""

STRUCTURED_OUTPUT_PROMPT = """You are a precise and helpful corporate assistant. Answer based on the document extracts provided below.

IMPORTANT INSTRUCTIONS:
1. Synthesize information from ALL relevant extracts to answer the question.
2. Ground every claim in the CONTEXT. NEVER invent facts, figures, names, or definitions that are not present.
3. DEFINITIONS: When the user asks "what is X?" and the CONTEXT defines X explicitly, reuse that definition faithfully. Keep distinct concepts (system, framework, module) separate — do not merge them into an invented name.
4. CONVERSATION HISTORY IS NOT A SOURCE: the history block is only conversational context and must never be quoted or used as a factual source — only CONTEXT extracts may be used.
5. ONLY if the information is completely missing from all extracts, reply: "I cannot find this information in the available documents."
6. Respond ONLY in valid JSON with this exact structure:
   {
     "answer": "your complete answer here",
     "citations": [
       {"chunk_id": "chunk identifier", "quote": "exact source phrase"}
     ]
   }
7. Do NOT include any text outside the JSON.
8. Citations must be exact phrases taken from the extracts.

CONTEXT:
{context}"""


class OllamaLLM:
    """
    Client to interact with the local Ollama API.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

        # M2 — état de santé observé (None = jamais vérifié)
        self.last_health: bool | None = None
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    # -- M2 : health check ---------------------------------------------------

    def health_check(self, timeout: float = HEALTH_CHECK_TIMEOUT) -> bool:
        """Vérifie la disponibilité d'Ollama (GET /api/tags).

        Mets à jour ``last_health`` et gère le circuit breaker : après
        ``CIRCUIT_BREAKER_FAILURES`` échecs consécutifs, le circuit est ouvert
        pendant ``CIRCUIT_BREAKER_COOLDOWN`` secondes ; les générations sont
        alors court-circuitées au lieu d'attendre un timeout réseau.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
            ok = resp.status_code == 200
        except requests.RequestException:
            ok = False

        self.last_health = ok
        if ok:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0
        else:
            self._consecutive_failures += 1
            if self._consecutive_failures >= CIRCUIT_BREAKER_FAILURES:
                self._circuit_open_until = time.time() + CIRCUIT_BREAKER_COOLDOWN
                logger.error(
                    "Circuit breaker ouvert (%d échecs consécutifs) pour %.0fs",
                    self._consecutive_failures, CIRCUIT_BREAKER_COOLDOWN,
                )
        return ok

    def _register_failure(self) -> None:
        """Enregistre un échec d'appel et déclenche éventuellement le breaker."""
        self.health_check(timeout=HEALTH_CHECK_TIMEOUT)

    @property
    def circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    # ------------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        formatted_context: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict:
        """
        Construit les messages (System + User) pour l'API Ollama.
        Prepends the last N conversation exchanges before the context block.
        """
        prompt_template = STRUCTURED_OUTPUT_PROMPT if USE_STRUCTURED_OUTPUT else SYSTEM_PROMPT
        system_content = prompt_template.format(context=formatted_context)

        # Build conversation prefix if history provided
        history_block = ""
        if conversation_history:
            window = conversation_history[-CONVERSATION_WINDOW:]
            lines = []
            for msg in window:
                role_label = "User" if msg.get("role") == "user" else "Assistant"
                lines.append(f"[{role_label}]: {msg.get('content', '')}")
            history_block = "\nHISTORIQUE DE CONVERSATION:\n" + "\n".join(lines) + "\n---\n"

        user_content = f"{history_block}QUESTION:\n{query}\n\nREPONSE:\n"

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "options": {
                "temperature": self.temperature,
            },
            "stream": True,
        }

    def generate_stream(
        self,
        query: str,
        formatted_context: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> Iterator[str]:
        """
        Envoie le prompt a Ollama et retourne un itérateur pour streamer la reponse.
        """
        payload = self.build_prompt(query, formatted_context, conversation_history)
        url = f"{self.base_url}/api/chat"

        logger.info("Envoi de la requete au modele %s (temp=%.2f) sur %s", self.model_name, self.temperature, url)

        if self.circuit_open:
            msg = ("Erreur : Ollama est actuellement indisponible (trop d'échecs récents). "
                   "Réessayez dans quelques instants.")
            logger.error(msg)
            yield msg
            return

        try:
            with requests.post(url, json=payload, stream=True, timeout=TIMEOUT_SECONDS) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        # Iter_lines renvoie des bytes ; le décodage strict en UTF-8
                        # de json.loads échoue (UnicodeDecodeError) quand un chunk
                        # Ollama coupe une séquence UTF-8 multi-octets (accents
                        # français). On décode en douceur avant le parsing.
                        try:
                            text = line.decode("utf-8", errors="replace")
                            chunk = json.loads(text)
                        except json.JSONDecodeError:
                            logger.warning("Ligne de flux Ollama mal formée, ignorée.")
                            continue
                        if "message" in chunk and "content" in chunk["message"]:
                            yield chunk["message"]["content"]

            self.health_check(timeout=HEALTH_CHECK_TIMEOUT)

        except requests.exceptions.ConnectionError as e:
            logger.error("Erreur de connexion a Ollama : %s", e)
            self._register_failure()
            yield f"Erreur : Impossible de se connecter a Ollama sur {self.base_url}. Le service est-il demarre ?"
        except requests.exceptions.Timeout as e:
            logger.error("Timeout de la requete Ollama : %s", e)
            self._register_failure()
            yield "Erreur : La requete vers le LLM a expire."
        except requests.exceptions.RequestException as e:
            logger.error("Erreur lors de la requete Ollama : %s", e)
            self._register_failure()
            yield f"Erreur de communication avec le LLM : {e}"

    def generate(
        self,
        query: str,
        formatted_context: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Genere la reponse complete d'un coup (bloquant).
        """
        full_response = []
        for chunk in self.generate_stream(query, formatted_context, conversation_history):
            full_response.append(chunk)
        return "".join(full_response)

    @staticmethod
    def parse_structured_response(raw_text: str) -> dict[str, Any]:
        """
        Try to parse a JSON structured response from the LLM.
        Returns {"answer": str, "citations": list[dict]} on success,
        or {"answer": raw_text, "citations": []} if parsing fails.
        """
        text = raw_text.strip()
        # Try to extract JSON from the response (LLM may wrap it in markdown)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "answer" in data:
                return {
                    "answer": data.get("answer", ""),
                    "citations": data.get("citations", []),
                }
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: return raw text as answer with no citations
        return {"answer": raw_text, "citations": []}

    def generate_hypothetical_answer(self, question: str) -> str:
        """
        Generate a hypothetical document-style answer for HyDE query expansion.
        The output is used as the search query instead of the raw question.
        """
        prompt = (
            "Generate a brief factual answer to this question as if you found it "
            "in a company document. Write 2-3 sentences in a formal tone. "
            f"Question: {question}"
        )
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.3},
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", question)
        except Exception as e:
            logger.warning("HyDE generation failed: %s. Using raw question.", e)
            return question

    def generate_sql(self, question: str, schema_text: str) -> str:
        """Generate a single read-only SQLite query for ``question`` given
        ``schema_text`` (see ``sql_query_engine.format_schema_for_prompt``).

        Non-streaming and temperature=0 (deterministic): unlike free-text
        generation, SQL has exactly one correct shape for a given question,
        so there is no upside to sampling and every downside (unparseable
        or unsafe output). The caller (``SQLQueryEngine.answer``) is
        responsible for validating and executing the returned string —
        this method only talks to the model.
        """
        system_content = SQL_GENERATION_SYSTEM_PROMPT.format(schema=schema_text)
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nSQL query:"},
            ],
            "options": {"temperature": 0.0},
            "stream": False,
        }

        if self.circuit_open:
            logger.error("Circuit breaker open — skipping SQL generation call.")
            return ""

        try:
            resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            self.health_check(timeout=HEALTH_CHECK_TIMEOUT)
            return data.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error("SQL generation request failed: %s", e)
            self._register_failure()
            return ""

    def generate_routed_sql(self, question: str, schemas: dict[str, str]) -> str:
        """Like ``generate_sql``, but for the multi-database case: ``schemas``
        maps each ingested database's filename to its schema text (see
        ``sql_query_engine.format_schema_for_prompt``). The model is shown
        every schema at once and must both pick the right database and
        write the query, returned as raw ``"DATABASE: ...\\nSQL: ..."`` text
        for ``sql_query_engine.parse_routed_response`` to parse — kept as a
        separate method (rather than overloading ``generate_sql``) so each
        has one obvious response shape and can be tested/stubbed in
        isolation.
        """
        schemas_block = "\n\n".join(
            f"--- Database: {name} ---\n{text}" for name, text in schemas.items()
        )
        system_content = SQL_ROUTING_SYSTEM_PROMPT.format(schemas=schemas_block)
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Question: {question}\n\nAnswer:"},
            ],
            "options": {"temperature": 0.0},
            "stream": False,
        }

        if self.circuit_open:
            logger.error("Circuit breaker open — skipping routed SQL generation call.")
            return ""

        try:
            resp = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
            resp.raise_for_status()
            data = resp.json()
            self.health_check(timeout=HEALTH_CHECK_TIMEOUT)
            return data.get("message", {}).get("content", "").strip()
        except requests.exceptions.RequestException as e:
            logger.error("Routed SQL generation request failed: %s", e)
            self._register_failure()
            return ""
