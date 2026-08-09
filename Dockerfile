# ==============================================================================
#  DOCKERFILE — RAG Souverain
#  Multi-stage build : embedding model pré-intégré, image runtime légère
#  Sprint 8 : Déploiement & Souveraineté
# ==============================================================================

# ── Stage 1 : Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

LABEL stage="builder"
LABEL description="RAG Souverain — Construction des dépendances et cache des modèles"

# Éviter les dialogs interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV TOKENIZERS_PARALLELISM=false

# Installer les dépendances système minimales (compilation éventuelle)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt /tmp/requirements.txt

# Installer torch (timeout long pour le gros download)
RUN pip install --no-cache-dir --default-timeout=600 torch --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances Python
RUN pip install --no-cache-dir --default-timeout=300 -r /tmp/requirements.txt

# ── Pre-telechargement du modele d'embedding (bake in image) ──
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); print(f'Modele embedding charge : {len(model.encode(\"test\"))} dimensions')"

# Pre-telechargement du modele CrossEncoder pour le reranking
RUN python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('CrossEncoder reranker pre-charge')"

# Nettoyage final
RUN apt-get purge -y gcc build-essential && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*


# ── Stage 2 : Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL description="RAG Souverain — Interface Streamlit + backend RAG"
LABEL version="2.0.0"
LABEL maintainer="Projet RAG Souverain — Sprint 8"

# Éviter les dialogs interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Installer iptables pour la souveraineté réseau
RUN apt-get update && apt-get install -y --no-install-recommends iptables && \
    rm -rf /var/lib/apt/lists/*

# Variables d'environnement par défaut (surchargeables via .env ou docker-compose)
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV OLLAMA_MODEL=mistral
ENV CHROMA_PERSIST_DIR=/app/data/chroma
ENV CHROMA_COLLECTION=rag_documents
ENV TOP_K=4
ENV PORT=8501
ENV TOKENIZERS_PARALLELISM=false
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
ENV HF_HOME=/app/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface
ENV LLM_TEMPERATURE=0.1

# Créer un utilisateur non-root pour la sécurité
RUN groupadd -r raguser && useradd -r -g raguser -m -d /home/raguser -s /bin/bash raguser

# Créer les répertoires de l'application
RUN mkdir -p /app/data/chroma /app/data/documents /app/data /home/raguser/.cache && \
    chown -R raguser:raguser /app /home/raguser

# Copier les dépendances Python depuis le builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copier le cache des modèles HuggingFace (embedding pré-téléchargé)
COPY --from=builder /root/.cache /app/.cache
RUN chown -R raguser:raguser /app/.cache

# Définir le répertoire de travail
WORKDIR /app

# Copier le code source de l'application
COPY --chown=raguser:raguser app/ /app/app/
COPY --chown=raguser:raguser data/documents/ /app/data/documents/
COPY --chown=raguser:raguser data/test_database.db /app/data/test_database.db

# Copier et configurer l'entrypoint de souveraineté réseau
COPY --chown=raguser:raguser entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Exposer le port Streamlit
EXPOSE 8501

# Passer à l'utilisateur non-root
USER raguser

# Point d'entrée : iptables puis Streamlit
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
