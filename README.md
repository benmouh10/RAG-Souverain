# RAG Souverain

Système RAG **100 % local et isolé** : interface Streamlit, génération via **Ollama**
(Mistral 7B), embeddings **sentence-transformers** hors-ligne et stockage vectoriel
**ChromaDB** persistant. Aucune donnée ne quitte la machine.

## Fonctionnalités

- **Authentification & RBAC** : deux rôles — `admin` et `user` — avec navigation et
  actions filtrées par rôle.
- **Chat RAG** : réponses sourcées (fichier, page, score), détection des salutations,
  historique persistant, export PDF de la conversation.
- **Indexation multi-fichiers** : PDF (natif + OCR), TXT, MD, CSV et SQLite, avec
  découpage adaptatif, versionnage et purge.
- **Paramètres système** : modèle LLM, température, `top_k`, taille de chunk.
- **Évaluation & audit** : pipeline Ragas, journal des requêtes, isolation réseau.
- **Performance** : métriques d'usage et temps de réponse.

## Stack

| Brique | Choix |
|---|---|
| UI | Streamlit 1.42 |
| LLM | Ollama — Mistral (local) |
| Embeddings | sentence-transformers (hors-ligne) |
| Stockage vectoriel | ChromaDB (HNSW, similarité cosinus) |
| Ingestion | pdfplumber, PyMuPDF, Tesseract OCR |
| Persistance | SQLite (utilisateurs, historique, versions) |
| Déploiement | Docker Compose + iptables (souveraineté réseau) |

## Structure

```
rag-souverain/
├── app/
│   ├── streamlit_app.py        # Point d'entrée Streamlit (navigation, session)
│   ├── app_ui.py               # Helpers UI, moteur RAG, helpers d'ingestion
│   ├── css.py                  # Styles (thème clair) et injection
│   ├── session.py              # Session state + registre des pages RBAC
│   ├── database.py             # SQLite : utilisateurs, versions, historique
│   ├── observability.py        # Journal des requêtes
│   ├── ingestion/              # Parser (PDF/TXT/MD/SQL), chunker, watcher
│   ├── indexing/               # ChromaVectorStore + embedding local
│   ├── retrieval/              # Retriever (recherche sémantique)
│   ├── generation/             # Client Ollama
│   ├── evaluation/             # Pipeline d'évaluation Ragas
│   └── pages/                  # Écrans (auth, chat, indexation, etc.)
├── data/                       # Chroma, documents, uploads, bases SQLite
├── tests/                      # Suite pytest (105 unit + 14 integration)
├── requirements.txt
├── pytest.ini
├── Dockerfile · docker-compose.yml · entrypoint.sh
└── CONCEPTION_STUDY.md         # Conception détaillée
```

## Démarrage rapide (dev local)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Prérequis : [Ollama](https://ollama.com) avec le modèle `mistral` chargé, et le modèle
d'embeddings `paraphrase-multilingual-MiniLM-L12-v2` disponible dans le cache
huggingface local (mode hors-ligne).

Lancer l'interface :

```bash
streamlit run app/streamlit_app.py
```

Compte par défaut : `admin` / `admin` (à changer en production).

## Tests

Le run par défaut n'exécute **que les tests unitaires** (les tests d'intégration sont
marqués `@pytest.mark.integration` et exclus via `pytest.ini` → `addopts = -m "not integration"`) :

```bash
python -m pytest tests/ -q
# Attendu : 105 passed, 14 deselected
```

Tests d'intégration (exigent Ollama démarré + modèle d'embeddings réel) :

```bash
ollama serve &                       # si Ollama n'est pas déjà démarré
python -m pytest tests/ -m integration -v
# Attendu : 14 passed (9 souveraineté réseau + 5 pipeline/RAG/embeddings)
```

Suite complète (unitaires + intégration) :

```bash
python -m pytest tests/ -q && python -m pytest tests/ -m integration -q
# 105 passed + 14 passed
```

### Validation dans le conteneur (NFR-01 — souveraineté réseau)

Les tests sont embarqués dans l'image Docker (exécutables en isolation réseau) :

```bash
docker compose up --build -d
docker exec rag-souverain python -m pytest tests/ -m "not integration" -q   # 105 passed
docker exec rag-souverain python -m pytest tests/ -m integration -q         # 14 passed
docker exec rag-souverain python tests/test_network_sovereignty.py          # audit live : 9/9 PASS
```

Démonstration devant jury (blocage Internet effectif) :

```bash
docker exec rag-souverain python -c "import urllib.request; print(urllib.request.urlopen('https://google.com', timeout=3).status)"
# Attendu : URLError — Temporary failure in name resolution  →  aucune donnée ne sort
```

## Documentation

- `DEPLOYMENT.md` — déploiement Docker et souveraineté réseau.
- `TESTING_GUIDE.md` — guide des tests d'acceptation.
