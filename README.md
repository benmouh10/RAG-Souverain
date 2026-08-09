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
├── tests/                      # Suite pytest (53 tests)
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

```bash
python -m pytest tests/ -q --ignore=tests/test_pipeline_complet.py --ignore=tests/test_rag.py
```

Les fichiers `test_pipeline_complet.py` et `test_rag.py` sont des tests d'intégration
(exigent Ollama / infrastructure réelle) et sont marqués `@pytest.mark.integration`.

## Documentation

- `DEPLOYMENT.md` — déploiement Docker et souveraineté réseau.
- `TESTING_GUIDE.md` — guide des tests d'acceptation.
