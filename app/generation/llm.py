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
TIMEOUT_SECONDS = 120
CONVERSATION_WINDOW = int(os.getenv("CONVERSATION_WINDOW", "3"))
USE_STRUCTURED_OUTPUT = os.getenv("USE_STRUCTURED_OUTPUT", "false").lower() in ("true", "1", "yes")

# M2 — Health check & circuit breaker
HEALTH_CHECK_TIMEOUT = 10.0
CIRCUIT_BREAKER_FAILURES = 3      # échecs consécutifs avant ouverture
CIRCUIT_BREAKER_COOLDOWN = 30.0   # durée d'ouverture (secondes)

SYSTEM_PROMPT = """Tu es un assistant d'entreprise precis et utile. Tu reponds en te basant sur les extraits de documents fournis ci-dessous.

INSTRUCTIONS IMPORTANTES :
1. Synthetise les informations provenant de TOUS les extraits pertinents pour repondre a la question.
2. Si l'information est contenue (meme partiellement) dans les extraits, fournis une reponse complete en combinant les informations disponibles.
3. Uniquement si l'information est completement absente de TOUS les extraits, reponds : "Je ne trouve pas cette information dans les documents disponibles."
4. Cite toujours la source de ton information (nom du fichier et page si disponible).
5. Ne reinvente pas d'informations, mais n'hesite pas a synthetiser et structurer ce qui est present dans les extraits.

CONTEXTE:
{context}"""

STRUCTURED_OUTPUT_PROMPT = """Tu es un assistant d'entreprise precis et utile. Tu reponds en te basant sur les extraits de documents fournis ci-dessous.

INSTRUCTIONS IMPORTANTES :
1. Synthetise les informations provenant de TOUS les extraits pertinents pour repondre a la question.
2. Si l'information est contenue (meme partiellement) dans les extraits, fournis une reponse complete en combinant les informations disponibles.
3. Uniquement si l'information est completement absente de TOUS les extraits, reponds : "Je ne trouve pas cette information dans les documents disponibles."
4. Tu dois repondre UNIQUEMENT en JSON valide avec cette structure exacte :
   {
     "answer": "ta reponse complete ici",
     "citations": [
       {"chunk_id": "identifiant du chunk", "quote": "phrase exacte source"}
     ]
   }
5. Ne ajoute AUCUN texte en dehors du JSON.
6. Les citations doivent etre des phrases exactes tirees des extraits.

CONTEXTE:
{context}"""


class OllamaLLM:
    """
    Client pour interagir avec l'API locale d'Ollama.
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
                        chunk = json.loads(line)
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
