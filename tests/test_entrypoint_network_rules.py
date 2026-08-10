"""Validation statique de la configuration de souveraineté réseau (NFR-01).

Sans Docker, les règles iptables ne peuvent pas être exécutées. Ces tests
vérifient donc la **structure** de la configuration :
  - entrypoint.sh applique un DROP OUTPUT par défaut (seul Ollama autorisé) ;
  - docker-compose.yml isole le réseau (DNS local 127.0.0.1, réseau dédié,
    cap_add NET_ADMIN, accès hôte limité à host.docker.internal).

La validation **live** (réelle application des règles + blocage) se fait dans
le conteneur : ``docker exec rag-souverain python tests/test_network_sovereignty.py``
(voir README, section « Souveraineté réseau »).
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = PROJECT_ROOT / "entrypoint.sh"
COMPOSE = PROJECT_ROOT / "docker-compose.yml"


def test_entrypoint_applies_default_output_drop():
    """Le trafic sortant est bloqué par défaut (aucune fuite réseau possible)."""
    script = ENTRYPOINT.read_text(encoding="utf-8")
    assert "iptables -A OUTPUT -j DROP" in script


def test_entrypoint_allows_loopback_and_established():
    """Loopback et connexions établies restent autorisés (services locaux)."""
    script = ENTRYPOINT.read_text(encoding="utf-8")
    assert "iptables -A OUTPUT -o lo -j ACCEPT" in script
    assert "ESTABLISHED,RELATED -j ACCEPT" in script


def test_entrypoint_allowlists_only_ollama_port():
    """Le seul trafic sortant autorisé est Ollama (hôte, port 11434)."""
    script = ENTRYPOINT.read_text(encoding="utf-8")
    assert "--dport 11434 -j ACCEPT" in script


def test_entrypoint_fallbacks_without_iptables():
    """Si iptables est absent, l'isolation DNS reste activée et on l'avertit."""
    script = ENTRYPOINT.read_text(encoding="utf-8")
    assert "AVERTISSEMENT" in script


def test_compose_forces_local_dns():
    """Le DNS externe est bloqué : seule la résolution locale répond."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "dns:" in compose
    assert "127.0.0.1" in compose


def test_compose_grants_net_admin_for_iptables():
    """Le conteneur a NET_ADMIN pour appliquer les règles iptables."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "cap_add:" in compose
    assert "NET_ADMIN" in compose


def test_compose_restricts_host_access_to_ollama():
    """host.docker.internal est limité au gateway (Ollama sur l'hôte)."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "host.docker.internal:host-gateway" in compose


def test_compose_uses_dedicated_network():
    """Les services vivent sur le réseau isolé rag_network."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "rag_network" in compose


def test_offline_hf_environment_enforced():
    """Les variables OFFLINE empêchent tout téléchargement au chargement."""
    compose = COMPOSE.read_text(encoding="utf-8")
    assert "TRANSFORMERS_OFFLINE=1" in compose
    assert "HF_HUB_OFFLINE=1" in compose
