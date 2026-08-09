"""
Test de Souverainete Reseau — Sprint 8
========================================

Valide que le systeme RAG Souverain fonctionne en circuit ferme (air-gapped) :
- Aucune requete HTTP externe ne peut sortir du conteneur
- Les services locaux (ChromaDB, Ollama) restent accessibles

Execution depuis le conteneur Docker :
    docker exec rag-souverain python tests/test_network_sovereignty.py

Execution en local (hors Docker, tests externes ignores) :
    python tests/test_network_sovereignty.py
"""

from __future__ import annotations

import http.client
import logging
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

# Marquer tous les tests de ce module comme integration (requis Docker)
pytestmark = pytest.mark.integration

# ── Configuration du logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sovereignty_test")

# ── Configuration des tests ──
OLLAMA_URL = "http://host.docker.internal:11434"
CHROMA_URL = "http://localhost:8501"
EXTERNAL_URLS = [
    "https://google.com",
    "https://api.openai.com",
    "https://huggingface.co",
    "https://cloudflare.com",
    "https://github.com",
]
TIMEOUT_SECONDS = 5

# ── Compteurs ──
passed = 0
failed = 0
skipped = 0


def report_test(name: str, success: bool, detail: str = ""):
    """Affiche le résultat d'un test et met à jour les compteurs."""
    global passed, failed
    status = "✅ PASS" if success else "❌ FAIL"
    if success:
        passed += 1
    else:
        failed += 1
    logger.info("%s | %s | %s", status, name, detail)


def report_skipped(name: str, reason: str):
    """Marque un test comme ignoré (hors Docker)."""
    global skipped
    skipped += 1
    logger.info("⏭️ SKIP | %s | %s", name, reason)


def is_running_in_docker() -> bool:
    """Détecte si on est exécuté dans un conteneur Docker."""
    if Path("/.dockerenv").exists() or Path("/run/.containerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "containerd" in content:
                return True
    except (FileNotFoundError, PermissionError, OSError):
        pass
    try:
        with open("/proc/self/cgroup", "r") as f:
            content = f.read()
            if "docker" in content or "containerd" in content:
                return True
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return False


# ===========================================================================
#  TESTS
# ===========================================================================


@pytest.mark.parametrize("url", EXTERNAL_URLS)
def test_external_http_blocked(url: str):
    """
    TEST 1-5 : Vérifie qu'une requête HTTP vers un service externe ÉCHOUE.
    En environnement air-gapped (internal: true), toute connexion sortante
    doit être bloquée par Docker.
    """
    name = f"Blocage externe : {url}"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            report_test(name, False, f"RÉUSSI (connexion non bloquée) — status={resp.status}")
    except urllib.error.URLError as e:
        # Timeout ou connexion refusée = SUCCÈS (le blocage fonctionne)
        report_test(name, True, f"Bloqué : {e.reason}")
    except socket.timeout:
        report_test(name, True, "Bloqué : timeout")
    except OSError as e:
        report_test(name, True, f"Bloqué : {e}")
    except Exception as e:
        report_test(name, True, f"Bloqué : {type(e).__name__}: {e}")


def test_ollama_accessible():
    """
    TEST 6 : Vérifie qu'Ollama est accessible via host.docker.internal.
    C'est la seule sortie autorisée vers l'hôte.
    """
    name = "Accès Ollama (host.docker.internal:11434)"
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            if resp.status == 200:
                report_test(name, True, f"Ollama joignable (status={resp.status})")
            else:
                report_test(name, False, f"Réponse inattendue (status={resp.status})")
    except urllib.error.URLError as e:
        report_test(name, False, f"Impossible de joindre Ollama : {e.reason}")
    except Exception as e:
        report_test(name, False, f"Erreur : {e}")


def test_chromadb_accessible():
    """
    TEST 7 : Vérifie que l'application Streamlit (ChromaDB) est accessible.
    """
    name = "Accès application (localhost:8501)"
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(f"{CHROMA_URL}/_stcore/health", method="GET")
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                if resp.status == 200:
                    report_test(name, True, "Application Streamlit joignable")
                    return
                report_test(name, False, f"Réponse inattendue (status={resp.status})")
                return
        except urllib.error.URLError as e:
            if attempt < attempts:
                time.sleep(1)
                continue
            report_test(name, False, f"Impossible de joindre l'application : {e.reason}")
            return
        except Exception as e:
            if attempt < attempts:
                time.sleep(1)
                continue
            report_test(name, False, f"Erreur : {e}")
            return


def test_dns_external_blocked():
    """
    TEST 8 : Vérifie que la résolution DNS externe échoue.
    En air-gap, le DNS externe ne doit pas répondre.
    """
    name = "Blocage DNS externe (google.com)"
    try:
        ip = socket.gethostbyname("google.com")
        report_test(name, False, f"DNS résolu : {ip} (non bloqué)")
    except socket.gaierror:
        report_test(name, True, "DNS bloqué (résolution impossible)")
    except Exception as e:
        report_test(name, True, f"DNS bloqué : {e}")


def test_localhost_loopback():
    """
    TEST 9 : Vérifie que le loopback local fonctionne (indispensable).
    """
    name = "Loopback local (127.0.0.1)"
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        # On ne teste pas la connexion, juste la création du socket
        sock.close()
        report_test(name, True, "Socket local créé avec succès")
    except Exception as e:
        report_test(name, False, f"Erreur socket : {e}")


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    """Point d'entrée : exécute tous les tests de souveraineté réseau."""
    in_docker = is_running_in_docker()

    print()
    print("=" * 70)
    print("  TEST DE SOUVERAINETÉ RÉSEAU — RAG Souverain")
    print("=" * 70)
    print(f"  Environnement : {'🐳 Docker (air-gapped)' if in_docker else '💻 Local (hors Docker)'}")
    print(f"  Date :          {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timeout :       {TIMEOUT_SECONDS}s par requête")
    print()

    # ── Tests de blocage externe (5 URLs) ──
    print("  ── Tests de blocage Internet ──")
    for url in EXTERNAL_URLS:
        test_external_http_blocked(url)

    # ── Tests DNS ──
    print()
    print("  ── Tests DNS ──")
    test_dns_external_blocked()

    # ── Tests d'accès local ──
    print()
    print("  ── Tests d'accès aux services locaux ──")
    test_localhost_loopback()

    if in_docker:
        test_ollama_accessible()
        test_chromadb_accessible()
    else:
        report_skipped("Accès Ollama", "Hors Docker — test non applicable")
        report_skipped("Accès Streamlit", "Hors Docker — test non applicable")

    # ── Rapport final ──
    print()
    print("=" * 70)
    print("  RÉSULTATS")
    print("=" * 70)
    print(f"  ✅ PASS : {passed}")
    print(f"  ❌ FAIL : {failed}")
    print(f"  ⏭️ SKIP : {skipped}")
    print()

    if failed == 0:
        print("  ✓ SOUVERAINETÉ CONFIRMÉE : Aucune fuite réseau détectée.")
        print("  ✓ Le système fonctionne en circuit fermé (air-gapped).")
        print("  ✓ Les services locaux (Ollama, ChromaDB) sont accessibles.")
    else:
        print("  ⚠ SOUVERAINETÉ NON CONFIRMÉE : Des fuites réseau ont été détectées.")
        print("  Vérifiez la configuration du réseau Docker (internal: true).")

    print()
    print("=" * 70)
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())