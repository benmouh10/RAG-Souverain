#!/bin/bash
# ==============================================================================
#  ENTRYPOINT — RAG Souverain
#  Applique les règles iptables pour la souveraineté réseau,
#  puis lance Streamlit.
# ==============================================================================

set -e

echo "[entrypoint] Application des règles de souveraineté réseau..."

# Autoriser le trafic loopback
iptables -A OUTPUT -o lo -j ACCEPT 2>/dev/null || true

# Autoriser les connexions établies (réponses aux requêtes sortantes)
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT 2>/dev/null || true

# Autoriser Ollama sur l'hôte (host.docker.internal = 172.17.0.1 ou via /etc/hosts)
# Résoudre host.docker.internal et autoriser le trafic vers ce seul IP
OLLAMA_IP=$(getent hosts host.docker.internal | awk '{ print $1 }' | head -1)
if [ -n "$OLLAMA_IP" ]; then
    iptables -A OUTPUT -d "$OLLAMA_IP" -p tcp --dport 11434 -j ACCEPT 2>/dev/null || true
    echo "[entrypoint] Ollama autorisé vers $OLLAMA_IP:11434"
fi

# Bloquer tout autre trafic sortant
iptables -A OUTPUT -j DROP 2>/dev/null || echo "[entrypoint] AVERTISSEMENT: iptables non disponible — souveraineté non appliquée via iptables (DNS isolation active)"

echo "[entrypoint] Démarrage de Streamlit..."
echo "[entrypoint] Interface disponible sur http://localhost:8501"

exec "$@"
