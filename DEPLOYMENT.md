# Déploiement & DevOps — RAG Souverain

> **Documentation de déploiement** — Sprint 8 (Finalisation & Déploiement)
>
> Ce fichier sert de section « Déploiement et DevOps » pour le rapport de stage.
> Il décrit l'architecture réseau des conteneurs, la procédure de lancement,
> la validation de la souveraineté et les procédures de dépannage.

---

## Table des matières

1. [Architecture réseau des conteneurs](#1-architecture-réseau-des-conteneurs)
2. [Prérequis](#2-prérequis)
3. [Démarrage en une commande](#3-démarrage-en-une-commande)
4. [Volumes et persistance des données](#4-volumes-et-persistance-des-données)
5. [Vérification du déploiement](#5-vérification-du-déploiement)
6. [Validation de la souveraineté réseau](#6-validation-de-la-souveraineté-réseau)
7. [Procédure de vérification des logs](#7-procédure-de-vérification-des-logs)
8. [Dépannage](#8-dépannage)
9. [Architecture des fichiers](#9-architecture-des-fichiers)

---

## 1. Architecture réseau des conteneurs

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                        MACHINE HÔTE                                 │
 │                                                                      │
 │  ┌──────────────────────────────────────────────┐                    │
 │  │         CONTENEUR DOCKER (rag-souverain)       │                    │
 │  │                                                │                    │
 │  │  ┌──────────┐    ┌───────────┐    ┌────────┐  │                    │
 │  │  │ Streamlit │───▶│ Retriever │───▶│ChromaDB│  │                    │
 │  │  │ (port 8501)│   │           │    │ (local)│  │                    │
 │  │  └──────────┘    └─────┬─────┘    └────────┘  │                    │
 │  │                        │                       │                    │
 │  │                        ▼                       │                    │
 │  │                  ┌──────────┐                  │                    │
 │  │                  │ OllamaLLM │                  │                    │
 │  │                  └────┬─────┘                  │                    │
 │  └───────────────────────┼────────────────────────┘                    │
 │                          │                                            │
 │                    ┌─────┴──────┐                                     │
 │                    │ Ollama hôte│                                     │
 │                    │ (port 11434)│                                     │
 │                    └────────────┘                                     │
 │                                                                      │
 │  Réseau Docker : rag_souverain_network (bridge, internal: true)       │
 │  → Aucune requête Internet ne peut sortir du conteneur               │
 │  → Seul host.docker.internal:11434 (Ollama) est accessible            │
 └─────────────────────────────────────────────────────────────────────┘
```

### Points clés de l'architecture

| Élément | Rôle | Détail technique |
|---|---|---|
| **Streamlit** | Interface utilisateur (port 8501) | Frontend exposé sur `localhost:8501` |
| **ChromaDB** | Base vectorielle locale | Intégrée en Python (pas de serveur séparé) |
| **Retriever** | Recherche sémantique | Interroge ChromaDB par similarité cosinus |
| **OllamaLLM** | Génération de réponse | Client HTTP vers Ollama sur l'hôte |
| **Ollama (hôte)** | LLM Mistral 7B | Tourne nativement avec accès GPU |

### Sécurité réseau

- **Réseau `rag_souverain_network`** : réseau bridge Docker isolé avec `internal: true`
- **Aucun accès Internet** : impossible de joindre des API externes (OpenAI, HuggingFace, etc.)
- **Seule exception** : `host.docker.internal:11434` pour communiquer avec Ollama sur l'hôte
- **Données en transit** : tout reste en local, à l'intérieur de la machine hôte

---

## 2. Prérequis

Avant de déployer le système, assurez-vous d'avoir installé les éléments suivants :

### 2.1 Docker Desktop

- **Windows** : [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
- **Linux** : Docker Engine 24+ et Docker Compose plugin
- **macOS** : [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)

Vérification :
```bash
docker --version
docker compose version
```

### 2.2 Ollama (sur la machine hôte)

Ollama doit être installé **sur la machine hôte** (pas dans le conteneur Docker) pour bénéficier de l'accélération GPU native.

```bash
# Installation (Linux / macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Téléchargement du modèle
ollama pull mistral

# Vérification
ollama list
```

> **Note Windows** : Téléchargez l'installateur depuis [ollama.com/download](https://ollama.com/download) et lancez `ollama pull mistral` dans un terminal.

### 2.3 Modèle d'embedding

Le modèle `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` est **pré-intégré dans l'image Docker** (étape de build). Aucun téléchargement supplémentaire n'est nécessaire au runtime.

---

## 3. Démarrage en une commande

### 3.1 Construction et lancement

```bash
# Depuis la racine du projet
docker compose up --build
```

Cette commande :
1. Construit l'image Docker (téléchargement des dépendances + pré-chargement du modèle d'embedding)
2. Monte les volumes persistants (ChromaDB, documents, cache HuggingFace)
3. Crée le réseau isolé `rag_souverain_network`
4. Lance le conteneur `rag-souverain`
5. Expose Streamlit sur `http://localhost:8501`

### 3.2 Démarrage en arrière-plan

```bash
docker compose up --build -d
```

### 3.3 Arrêt

```bash
# Arrêter les services
docker compose down

# Arrêter et supprimer les volumes (⚠️ supprime la base ChromaDB)
docker compose down -v
```

### 3.4 Reconstruction après modifications

```bash
docker compose up --build
```

---

## 4. Volumes et persistance des données

### 4.1 Volumes Docker

| Volume | Point de montage | Contenu | Persistance |
|---|---|---|---|
| `rag_chroma_data` | `/app/data/chroma` | Base vectorielle ChromaDB (index + embeddings) | ✅ Survit aux redémarrages |
| `rag_hf_cache` | `/root/.cache/huggingface` | Cache du modèle d'embedding | ✅ Évite le re-téléchargement |

### 4.2 Montages locaux (bind mounts)

| Fichier/Dossier hôte | Point de montage | Mode | Contenu |
|---|---|---|---|
| `./data/documents/` | `/app/data/documents` | Lecture seule (ro) | Documents source (PDF, TXT, MD) |
| `./data/test_database.db` | `/app/data/test_database.db` | Lecture seule (ro) | Base SQL de test |

### 4.3 Ajout de documents sans reconstruire l'image

Placez les fichiers dans `./data/documents/` sur votre machine hôte. Ils seront automatiquement disponibles dans le conteneur au prochain démarrage (ou via la fonction d'ingestion de l'interface Streamlit).

---

## 5. Vérification du déploiement

### 5.1 Vérifier que le conteneur tourne

```bash
docker ps

# Exemple de sortie attendue :
# CONTAINER ID   IMAGE              COMMAND                  CREATED         STATUS                   PORTS                    NAMES
# abc123def456   rag-souverain-rag  "streamlit run app/a…"   2 minutes ago   Up 2 minutes (healthy)   0.0.0.0:8501->8501/tcp   rag-souverain
```

### 5.2 Vérifier le healthcheck

```bash
docker inspect rag-souverain --format='{{.State.Health.Status}}'
# Doit retourner : healthy
```

### 5.3 Accéder à l'interface web

Ouvrez [http://localhost:8501](http://localhost:8501) dans votre navigateur.

Vous devriez voir la **Console d'Administration RAG Souverain** avec :
- Le statut des services (ChromaDB, Ollama)
- Le nombre de chunks indexés
- L'interface de chat pour poser des questions

### 5.4 Vérifier les logs en temps réel

```bash
# Logs en continu
docker compose logs -f

# Dernières 50 lignes
docker compose logs --tail=50

# Logs d'un service spécifique
docker logs rag-souverain
```

---

## 6. Validation de la souveraineté réseau

### 6.1 Exécution du script de test

Une fois le conteneur démarré, exécutez le script de test de souveraineté :

```bash
docker exec rag-souverain python tests/test_network_sovereignty.py
```

### 6.2 Résultat attendu

```
======================================================================
  TEST DE SOUVERAINETÉ RÉSEAU — RAG Souverain
======================================================================
  Environnement : 🐳 Docker (air-gapped)
  Date :          2026-07-02 12:30:00
  Timeout :       5s par requête

  ── Tests de blocage Internet ──
  ✅ PASS | Blocage externe : https://google.com          | Bloqué : timeout
  ✅ PASS | Blocage externe : https://api.openai.com      | Bloqué : timeout
  ✅ PASS | Blocage externe : https://huggingface.co      | Bloqué : timeout
  ✅ PASS | Blocage externe : https://cloudflare.com      | Bloqué : timeout
  ✅ PASS | Blocage externe : https://github.com          | Bloqué : timeout

  ── Tests DNS ──
  ✅ PASS | Blocage DNS externe (google.com)              | DNS bloqué

  ── Tests d'accès aux services locaux ──
  ✅ PASS | Loopback local (127.0.0.1)                    | Socket local créé
  ✅ PASS | Accès Ollama (host.docker.internal:11434)     | Ollama joignable
  ✅ PASS | Accès application (localhost:8501)             | Application OK

======================================================================
  RÉSULTATS
======================================================================
  ✅ PASS : 9
  ❌ FAIL : 0
  ⏭️ SKIP : 0

  ✓ SOUVERAINETÉ CONFIRMÉE : Aucune fuite réseau détectée.
  ✓ Le système fonctionne en circuit fermé (air-gapped).
  ✓ Les services locaux (Ollama, ChromaDB) sont accessibles.
======================================================================
```

### 6.3 Interprétation des résultats

| Condition | Résultat | Interprétation |
|---|---|---|
| **Tous les tests PASS** | ✅ Souveraineté confirmée | Le système est totalement isolé et souverain |
| **Un test FAIL** | ⚠ Fuite détectée | Vérifier la configuration réseau Docker (`internal: true`) |
| **Ollama FAIL** | ⚠ Service hôte indisponible | Vérifier qu'Ollama tourne sur l'hôte (`ollama serve`) |

---

## 7. Procédure de vérification des logs

### 7.1 Logs de l'application Streamlit

```bash
# Voir les logs de l'application
docker logs rag-souverain

# Suivre en temps réel
docker logs -f rag-souverain
```

### 7.2 Vérifier le pipeline RAG

```bash
# Tester le retrieval sans LLM
docker exec rag-souverain python -c "
from app.indexing.vector_store import ChromaVectorStore
vs = ChromaVectorStore(persist_directory='/app/data/chroma')
vs.initialize_collection()
print(f'Chunks indexés : {vs.count()}')
"

# Tester l'évaluation RAGAS
docker exec rag-souverain python -m app.evaluation.eval_pipeline
```

### 7.3 Vérifier la configuration réseau

```bash
# Voir les réseaux Docker
docker network ls

# Inspecter le réseau souverain
docker network inspect rag_souverain_network

# Vérifier qu'aucun conteneur n'a accès à Internet
docker exec rag-souverain ping -c 1 google.com
# Doit échouer avec : ping: bad address 'google.com'
```

---

## 8. Dépannage

### 8.1 Le conteneur ne démarre pas

**Symptôme** : `docker compose up --build` échoue
**Solutions** :
```bash
# Vérifier les logs détaillés
docker compose logs

# Vérifier que le port 8501 n'est pas déjà utilisé
netstat -ano | findstr :8501

# Réinitialiser Docker
docker compose down
docker system prune -a
docker compose up --build
```

### 8.2 Ollama inaccessible depuis le conteneur

**Symptôme** : L'interface Streamlit affiche « Ollama OFF »
**Solutions** :
```bash
# 1. Vérifier qu'Ollama tourne sur l'hôte
ollama list

# 2. Vérifier que le service écoute sur le port 11434
curl http://localhost:11434/api/tags

# 3. Redémarrer Ollama
ollama serve

# 4. Redémarrer le conteneur
docker compose restart
```

### 8.3 ChromaDB vide après redémarrage

**Symptôme** : « 0 chunks indexés » dans l'interface
**Causes possibles** :
- Volume supprimé (`docker compose down -v`)
- Base vectorielle dans un état incohérent

**Solution** : Ré-indexer les documents
```bash
docker exec rag-souverain python -c "
from app.ingestion.parser import DocumentParser, SQLParser
from app.ingestion.chunker import AdaptiveChunker
from app.indexing.vector_store import ChromaVectorStore

# Parse + chunk + index
parser = DocumentParser(ocr_enabled=False)
documents = parser.parse_directory('/app/data/documents')
sql_parser = SQLParser('/app/data/test_database.db')
documents.append(sql_parser.to_parsed_documents())

chunker = AdaptiveChunker()
chunks = chunker.chunk_documents(documents)

vs = ChromaVectorStore(persist_directory='/app/data/chroma')
vs.initialize_collection(reset=True)
added = vs.add_chunks(chunks)
print(f'{added} chunks indexés')
"
```

### 8.4 Lenteur au premier chargement

**Symptôme** : L'interface met du temps à s'afficher la première fois
**Cause** : Chargement du modèle d'embedding (SentenceTransformer) au premier appel
**Solution** : Le modèle est pré-intégré dans l'image, mais le premier chargement en mémoire peut prendre quelques secondes. C'est normal.

### 8.5 Problèmes de mémoire

**Symptôme** : Le conteneur se tue avec OOMKilled
**Solution** : Ajuster les limites mémoire dans `docker-compose.yml`
```yaml
deploy:
  resources:
    limits:
      memory: 4G    # Augmenter si nécessaire
    reservations:
      memory: 1G
```

---

## 9. Architecture des fichiers

```
rag-souverain/
├── .env                          # Variables d'environnement
├── Dockerfile                    # Image multi-stage avec embedding pré-intégré
├── docker-compose.yml            # Orchestration air-gapped
├── DEPLOYMENT.md                 # 🡐 Ce fichier — Documentation de déploiement
├── requirements.txt              # Dépendances Python
├── config.yaml                   # Configuration du pipeline
│
├── app/
│   ├── app_ui.py                 # Interface Streamlit
│   ├── main.py                   # API FastAPI
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── eval_pipeline.py      # Pipeline d'évaluation RAGAS
│   ├── generation/
│   │   ├── __init__.py
│   │   └── llm.py                # Client Ollama LLM
│   ├── indexing/
│   │   ├── __init__.py
│   │   └── vector_store.py       # ChromaDB Vector Store
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py            # Chunking adaptatif
│   │   └── parser.py             # Parseur PDF / TXT / SQL
│   └── retrieval/
│       ├── __init__.py
│       └── retriever.py          # Recherche sémantique
│
├── data/
│   ├── chroma/                   # 🡐 Volume persistant ChromaDB
│   ├── documents/                # Documents source
│   │   ├── stage_si_RAG.pdf
│   │   ├── test_markdown.md
│   │   ├── test_text.txt
│   │   └── dummy_test.pdf
│   └── test_database.db          # Base SQL de test
│
└── tests/
    ├── test_network_sovereignty.py   # 🡐 Test de souveraineté réseau
    ├── test_chunker.py
    ├── test_parser.py
    ├── test_pipeline_complet.py
    ├── test_rag.py
    ├── test_retriever.py
    └── test_sql.py
```

---

## Annexe : Commandes utiles

```bash
# ── Déploiement ──
docker compose up --build -d        # Lancer en arrière-plan
docker compose up --build           # Lancer en premier plan
docker compose down                 # Arrêter
docker compose down -v              # Arrêter et supprimer les volumes
docker compose restart              # Redémarrer

# ── Inspection ──
docker ps                           # Liste des conteneurs
docker logs rag-souverain           # Logs du conteneur
docker logs -f rag-souverain        # Logs en temps réel
docker inspect rag-souverain        # Détails du conteneur
docker stats rag-souverain          # Statistiques (CPU, mémoire)

# ── Tests ──
docker exec rag-souverain python tests/test_network_sovereignty.py
docker exec rag-souverain python -m app.evaluation.eval_pipeline

# ── Accès shell ──
docker exec -it rag-souverain /bin/bash

# ── Nettoyage ──
docker system prune -a              # Supprimer toutes les images inutilisées
docker volume prune                 # Supprimer les volumes orphelins
```

---

> **Document généré le 2 juillet 2026 — Projet RAG Souverain (PFA 2025-2026)**
> *Architecture et déploiement : Mohamed — Stage SI*