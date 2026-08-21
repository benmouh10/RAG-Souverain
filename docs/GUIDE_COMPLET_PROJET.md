# DOCUMENTATION COMPLÈTE DU PROJET — RAG Souverain

> Guide de référence : chaque fichier, son rôle et toutes ses fonctions expliquées.
> Généré pour comprendre l'intégralité du projet.

---

## 1. VUE D'ENSEMBLE

**RAG Souverain** est un système de questions-réponses **100 % local** sur vos documents :
il ingère des fichiers (PDF, TXT, MD, CSV, SQL), les découpe en chunks, les vectorise,
les indexe dans ChromaDB, puis répond aux questions via un LLM local (Mistral via Ollama)
en citant ses sources. Aucune donnée ne quitte la machine.

```
Utilisateur → Interface Streamlit → RAGEngine → Retriever (recherche) → OllamaLLM (génération)
                                        ↓                                    ↑
                                  ChromaDB (index)                    réponse + sources
                                        ↑
                        Ingestion : Parser → Chunker → VectorStore
```

---

## 2. FICHIERS À LA RACINE

### `README.md`
Documentation d'accueil : fonctionnalités, stack technique, structure du projet,
démarrage rapide (dev local), commandes de tests (unitaires/intégration/conteneur),
et démonstration de la souveraineté réseau.

### `requirements.txt`
Liste des dépendances Python avec versions minimales :
- **UI** : streamlit
- **Ingestion** : pdfplumber, pymupdf, pytesseract, Pillow
- **Indexation/Retrieval** : sentence-transformers, chromadb, transformers, rank-bm25
- **LLM** : requests, langchain-openai, langchain-huggingface
- **Évaluation** : ragas, datasets
- **Tests** : pytest, pytest-cov
- **Export PDF** : fpdf2
- **Watcher** : watchdog

### `pytest.ini`
Configuration des tests :
- Déclare le marker `integration` (tests nécessitant Ollama/ChromaDB/modèle réel).
- `addopts = -m "not integration"` : le run par défaut n'exécute QUE les tests unitaires.
- Pour lancer les tests d'intégration : `python -m pytest tests/ -m integration`.

### `docker-compose.yml`
Orchestration de deux services :
1. **rag-app** (conteneur `rag-souverain`) :
   - Port 8501 exposé (Streamlit) ;
   - Volumes persistants : `data/chroma` (base vectorielle), `data/rag.db` (SQLite),
     `data/documents` (lecture seule), `data/evaluation_report.csv` (rapport d'évaluation) ;
   - Accès à Ollama hôte via `host.docker.internal:11434` ;
   - Variables hors-ligne Hugging Face (`TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1`) ;
   - DNS forcé sur `127.0.0.1` (bloque toute résolution externe) ;
   - Réseau dédié `rag_network`, healthcheck Streamlit, limite mémoire 2G ;
   - `cap_add: NET_ADMIN` : permission nécessaire pour appliquer les règles iptables.
2. **ingestion-watcher** (conteneur `rag-ingestion-watcher`) :
   - Exécute `python -m app.ingestion.watcher` ;
   - Surveille `data/documents` en continu et indexe automatiquement les nouveaux fichiers.

### `entrypoint.sh`
Script exécuté au démarrage du conteneur AVANT Streamlit :
applique les règles iptables de souveraineté réseau (`OUTPUT DROP` par défaut,
allowlist loopback + Ollama hôte uniquement), puis lance la commande CMD.

---

## 3. COUCHE PRÉSENTATION (Interface)

### `app/streamlit_app.py` — Point d'entrée de l'application
Le fichier lancé par `streamlit run`. Construit la navigation et le layout global.

| Fonction | Rôle |
|---|---|
| `render_hero_header()` | Affiche le bandeau d'en-tête de la page d'accueil. |
| `render_sidebar()` | Construit la barre latérale : logo, infos utilisateur, bouton déconnexion, compteur de documents indexés. |
| `render_top_nav()` | Barre de navigation horizontale filtrée par rôle (RBAC). Retourne la page sélectionnée. |
| `reset_chat()` | Réinitialise l'état de la conversation dans la session. |
| `render_floating_scroll_buttons()` | Boutons flottants haut/bas pour faire défiler le chat. |
| `main()` | Point d'entrée : injecte les styles, gère l'état de session, route vers la page demandée (login ou pages métier selon le rôle). |

### `app/session.py` — État de session & RBAC
Gère la session Streamlit (qui est connecté, quel rôle) et la liste des pages autorisées.

| Fonction | Rôle |
|---|---|
| `init_session_state()` | Initialise les clés de session (authentifié, page courante, etc.). |
| `sign_in(user_id, username, role)` | Enregistre l'utilisateur connecté dans la session. |
| `sign_out()` | Déconnecte et nettoie la session. |
| `get_role()` | Retourne le rôle courant (`admin`/`user`) ou None. |
| `is_admin()` | True si l'utilisateur courant est admin. |
| `require_admin()` | **Garde serveur (M1)** : bloque le rendu si pas admin (défense en profondeur au-delà du simple masquage UI). |
| `allowed_pages()` | Liste des pages accessibles selon le rôle (admin : 6 pages ; user : chat + consultation). |
| `set_current_page(page)` | Mémorise la page active. |

### `app/css.py` — Styles visuels

| Fonction | Rôle |
|---|---|
| `inject_styles()` | Injecte le CSS global (thème clair) dans la page Streamlit. |
| `hide_sidebar_style()` | CSS pour masquer/déguiser la sidebar quand nécessaire. |
| `sidebar_logo_html()` | HTML du logo affiché dans la sidebar. |
| `login_logo_html()` | HTML du logo affiché sur la page de connexion. |

### `app/app_ui.py` — Helpers partagés entre les pages
Boîte à outils centrale : chargements paresseux (lazy imports), construction des messages,
export PDF, ingestion de fichiers uploadés.

| Fonction | Rôle |
|---|---|
| `_load_rag_engine_class()` | Import différé de `RAGEngine` (allège le démarrage). |
| `_load_eval_helpers()` | Import différé de `run_evaluation` / `export_results`. |
| `_load_llm_class()` | Import différé de `OllamaLLM`. |
| `_load_ingestion_helpers()` | Import différé de `AdaptiveChunker` + `DocumentParser`. |
| `_load_database()` | Import différé de `Database`. |
| `_load_observability()` | Import différé de `QueryLogger`. |
| `class IngestError(Exception)` | Erreur métier lors d'une ingestion (fichier vide, non supporté…). |
| `class OllamaConnectionError(Exception)` | Erreur de connexion à Ollama. |
| `init_user_session(user_id)` | Prépare l'état de session lié à l'utilisateur (conversation courante…). |
| `build_chat_message(role, content, sources, citations)` | Fabrique un message de chat structuré (rôle, texte, sources citées). |
| `get_upload_widget_key()` / `reset_upload_widget_state()` | Gèrent la clé du widget d'upload pour pouvoir le réinitialiser après un upload réussi. |
| `build_source_payloads(results)` | Transforme les `SearchResult` en payloads affichables (fichier, page, score, extrait). |
| `_sanitize_for_pdf(text)` | Nettoie le texte pour fpdf2 (caractères non latin-1 remplacés). |
| `export_chat_to_pdf(messages, username)` | Génère le PDF de la conversation (fpdf2) : titre, date, messages, sources. Retourne les octets du PDF. |
| `init_rag_engine(force, model_name)` | Crée/récupère le `RAGEngine` en session (singleton par modèle). Gère les erreurs de connexion. |
| `get_chroma_count(rag)` | Nombre de chunks indexés (affiché dans la sidebar). |
| `delete_file_from_vector_store(filename, rag)` | Supprime tous les chunks d'un fichier de l'index. |
| `aggregate_sources(sources)` | Regroupe les sources identiques (même fichier+page) en additionnant les scores. |
| `compute_cosine_score(rag, query)` | Score cosinus entre la question et le meilleur chunk (utilisé pour l'affichage). |
| `ingest_uploaded_file(uploaded_file, rag)` | Ingestion complète d'un fichier uploadé : hash → parsing → chunking → versionnage → indexation. Retourne (chunks ajoutés, version). |
| `is_greeting(query)` | Détecte les salutations courtes (« bonjour », « salut »…) pour répondre sans recherche vectorielle. |
| `dedupe_sources(sources)` | Déduplique les sources par clé `file|page|content` (évite les doublons à l'affichage). |
| `truncate_snippet(content, max_chars)` | Raccourcit un extrait de source pour l'affichage. |
| `main()` | Point d'entrée si exécuté directement (lance Streamlit). |

---

## 4. COUCHE AUTHENTIFICATION & DONNÉES

### `app/database.py` — SQLite : utilisateurs, versions, historique
Toute la persistance relationnelle. Deux dataclasses + une classe façade.

**Fonctions module :**

| Fonction | Rôle |
|---|---|
| `_hash_password(password, salt)` | Hachage scrypt du mot de passe avec sel aléatoire. |
| `_verify_password(stored, password)` | Vérifie un mot de passe contre le hash stocké. |
| `_is_none_string(val)` | Détecte les chaînes « None » parasites (nettoyage BDD). |
| `_sanitize_val(val, default)` | Assainit une valeur avant insertion SQLite. |
| `_sanitize_sources(sources)` | Assainit la liste des sources stockée en JSON. |
| `compute_file_hash(data)` | Hash SHA-256 d'un fichier (détection des ré-indexations). |

**Classes :**

| Classe | Champs |
|---|---|
| `User` | user_id, username, password_hash, salt, role, created_at |
| `DocVersion` | doc_id, filename, file_hash, version, chunk_count, indexed_at |

**Classe `Database` — méthodes :**

*Utilisateurs :*
| Méthode | Rôle |
|---|---|
| `__init__(db_path)` | Ouvre/crée la base SQLite. |
| `_connect()` | Connexion avec accès par nom de colonne (row_factory). |
| `_init_db()` | Crée les tables (users, doc_versions, conversations, chat_messages) + migration automatique du versionnage global (suppression de user_id). |
| `create_user(username, password, role)` | Crée un compte (hash scrypt). Retourne None si le pseudo existe déjà. |
| `authenticate(username, password)` | Vérifie les identifiants. Retourne le User ou None. |
| `get_user_by_username(username)` / `get_user_by_id(user_id)` | Recherche d'un utilisateur. |
| `list_users()` | Tous les utilisateurs. |
| `change_password(user_id, new_password)` | Change le mot de passe (re-hash). |
| `delete_user(user_id)` | Supprime un compte. |

*Versionnage global des documents (C1/Mo4) :*
| Méthode | Rôle |
|---|---|
| `record_ingestion(filename, file_hash, chunk_count)` | Enregistre une nouvelle version d'un fichier (version = précédente + 1). La clé est le nom de fichier seul (partagé entre admins). |
| `get_latest_version(filename)` | Dernière version connue d'un fichier. |
| `get_file_hash(filename)` | Hash du dernier indexage (pour détecter un fichier inchangé). |
| `list_documents()` | Tous les documents indexés (nom, version, chunks, date). |
| `get_total_chunks()` | Somme des chunks de tous les documents. |

*Conversations & historique :*
| Méthode | Rôle |
|---|---|
| `create_conversation(user_id, title)` | Nouvelle conversation. Retourne son id. |
| `list_conversations(user_id)` | Conversations d'un utilisateur. |
| `rename_conversation(conv_id, title)` / `delete_conversation(conv_id)` | Gestion des conversations. |
| `save_chat_message(...)` | Persiste un message (question ou réponse) avec ses sources JSON. |
| `get_chat_history(user_id, conversation_id, limit)` | Relit l'historique (bridge session_state ↔ SQLite, C3). |
| `clear_chat_history(...)` | Efface l'historique. |
| `cleanup_old_messages(max_age_days)` | Purge des messages > 30 jours. |
| `cleanup_none_messages(user_id)` | Supprime les messages corrompus (« None »). |

### `app/pages/auth.py` — Page de connexion

| Fonction | Rôle |
|---|---|
| `render_login_page()` | Formulaire username/password → `db.authenticate()` → `sign_in()`. L'auto-inscription a été supprimée : la création de compte est réservée à l'admin via User Management. |

### `app/pages/users.py` — Gestion des comptes (admin)

| Fonction | Rôle |
|---|---|
| `render_users_page()` | Tableau des utilisateurs + formulaire de création (pseudo, mdp, rôle) + changement de mot de passe + suppression. Protégé par `require_admin()`. |

---

## 5. COUCHE INGESTION (Module 1)

### `app/ingestion/parser.py` — Extraction de texte multi-formats

**Classes de données :**

| Classe | Rôle |
|---|---|
| `ExtractionMethod(Enum)` | Méthode utilisée : PDF_NATIVE, PDF_OCR, TEXT, SQL. |
| `ParsedPage` | Une page extraite : contenu, numéro de page, méthode d'extraction. |
| `ParsedDocument` | Document complet : liste de pages + métadonnées. Propriétés : `is_empty()`, `source_file()`, `full_text()`, `page_count()`, `to_parsed_page()`. |
| `SQLRecord` | Un enregistrement SQL transformé en texte indexable. |

**Classe `SQLParser` :**

| Méthode | Rôle |
|---|---|
| `__init__(db_path)` | Ouvre la base SQLite à indexer. |
| `get_tables()` | Liste les tables. |
| `get_primary_key(table)` | Clé primaire d'une table. |
| `parse_table(table)` | Lit toutes les lignes d'une table → SQLRecords. |
| `parse_all_tables()` | Toutes les tables. |
| `to_parsed_documents()` | Convertit les enregistrements en ParsedDocument (un enregistrement ≈ une ligne de texte). |
| `close()` | Ferme la connexion. |

**Classe `DocumentParser` :**

| Méthode | Rôle |
|---|---|
| `parse_directory(directory)` | Parse tous les fichiers supportés d'un dossier. |
| `parse_file(file_path)` | Route vers le bon parseur selon l'extension (.pdf/.txt/.md/.csv). |
| `iter_pages(directory)` | Itère page par page (économie mémoire sur gros corpus). |
| `_parse_pdf(path)` | Stratégie PDF : essaie natif d'abord, OCR en secours page par page. |
| `_extract_pdf_native(path)` | Extraction native : pdfplumber puis PyMuPDF en fallback. |
| `_extract_with_pdfplumber(path)` | Extraction via pdfplumber (bon pour tableaux). |
| `_extract_with_pymupdf(path)` | Extraction via PyMuPDF (rapide). |
| `_extract_pdf_page_ocr(path, page_number)` | OCR Tesseract d'une page (PDF scanné). |
| `_parse_text_file(path)` | Lecture TXT/MD/CSV avec détection d'encodage. |
| `_read_text_with_fallback(path)` | Essaie UTF-8 puis encodages de repli. |
| `_needs_ocr(text)` | True si le texte extrait est trop vide (→ page scannée). |

### `app/ingestion/chunker.py` — Découpage adaptatif

**Classes :**

| Classe | Rôle |
|---|---|
| `ChunkStrategy(Enum)` | Stratégies : SECTION, SLIDING_WINDOW, TABLE, SQL. |
| `TextChunk` | Un chunk final : contenu + métadonnées (source_file, page_number, chunk_index, strategy, version). Méthode `to_metadata()`. |
| `AdaptiveChunker` | Le découpeur. Paramètres : taille cible (tokens), overlap, min_size. |

**Méthodes de `AdaptiveChunker` :**

| Méthode | Rôle |
|---|---|
| `chunk_documents(documents)` | Point d'entrée : découpe une liste de ParsedDocument. |
| `chunk_document(document)` | Découpe un document : choisit la stratégie par page. |
| `_chunk_page(page, start_index)` | Logique par page : tableau ? SQL ? structuré ? texte libre ? |
| `_is_table(text)` | Détecte un tableau (lignes avec séparateurs `|`). |
| `_chunk_by_table(page, start_index)` | Préserve chaque ligne du tableau intacte. |
| `_chunk_sql(page, start_index)` | Un enregistrement SQL = un chunk. |
| `_is_structured(text)` | Détecte une structure par titres (Markdown/numérotation). |
| `_chunk_by_sections(...)` | Découpe par sections/titres (overlap 0). |
| `_chunk_by_sliding_window(...)` | Fenêtre glissante de N tokens avec overlap (texte libre). |
| `_split_into_sections(text)` | Segmente le texte en (titre, contenu). |
| `_get_tokenizer()` / `_encode()` / `_decode()` / `_count_tokens()` | Tokenisation (transformers) pour compter les tokens réels. |

### `app/ingestion/watcher.py` — Ingestion continue (C5)
Processus séparé (2ᵉ conteneur) qui surveille `data/documents` et indexe automatiquement
tout nouveau fichier modifié. Périmètre volontairement restreint (non récursif).

| Fonction | Rôle |
|---|---|
| `_compute_file_hash(file_path)` | SHA-256 du fichier (déduplication). |
| `_init_hash_db()` | Petite base SQLite des fichiers déjà indexés (hash + date). |
| `_is_indexed(conn, path)` | Fichier déjà traité (hash identique) ? |
| `_is_changed(conn, path)` | Fichier modifié depuis la dernière ingestion ? |
| `_record_indexed(conn, path, chunks)` | Marque le fichier comme indexé. |
| `_should_ignore(path)` | Filtres : fichiers temporaires, cachés, extensions non supportées. |
| `_ingest_file(path, conn)` | Pipeline complet : parse → chunk → indexe → enregistre. |
| `_scan_and_ingest(conn)` | Scan initial du dossier au démarrage. |
| `class WatcherEventHandler` | Réagit aux événements watchdog (création/modification) avec debounce. |
| `main()` | Boucle principale : scan initial + observation continue. |

---

## 6. COUCHE INDEXATION (Module 3)

### `app/indexing/vector_store.py` — ChromaDB + embeddings locaux

**Fonctions module :**

| Fonction | Rôle |
|---|---|
| `_safe_meta(meta)` | Assainit les métadonnées pour ChromaDB (dict/list → JSON, None → ""). |
| `_resolve_local_sentence_transformers_path(model_name, cache_root)` | Trouve le snapshot LOCAL du modèle dans le cache HF (mode hors-ligne garanti, aucun téléchargement). |
| `_format_results(results)` | Convertit la réponse brute ChromaDB en liste de SearchResult. |

**Classe `LocalEmbeddingFunction` :**
Embeddings 100 % locaux (sentence-transformers, multilingue, 384 dimensions).

| Méthode | Rôle |
|---|---|
| `__init__(model_name)` | Charge le modèle depuis le cache local (OFFLINE). |
| `name()` | Nom du modèle. |
| `__call__(input)` | Embedding d'une liste de textes (batch). |
| `embed_query(query)` | Embedding d'une seule requête. |

**Dataclass `SearchResult` :**
Résultat de recherche normalisé : `id`, `content`, `score`, `metadata` (source_file, page_number, …).

**Classe `ChromaVectorStore` :**

| Méthode | Rôle |
|---|---|
| `__init__(persist_directory, collection_name, embedding_function)` | Configure le client persistant ChromaDB. |
| `initialize_collection(reset=False)` | Crée/ouvre la collection (HNSW, cosine). reset=True vide tout. |
| `collection()` | Accès direct à la collection sous-jacente. |
| `add_chunks(chunks, batch_size)` | Indexe des TextChunks : embed + upsert avec métadonnées assainies. Retourne le nombre ajouté. |
| `search(query, top_k, where, min_score)` | Recherche vectorielle pure avec filtre `where` optionnel. |
| `search_latest(query, top_k, max_version)` | Recherche restreinte à la DERNIÈRE version indexée (C4 : égalité stricte, pas $lte). |
| `get_latest_version_all()` | Version maximale présente dans toute la collection. |
| `get_latest_version(filename)` | Dernière version d'un fichier donné. |
| `count_by_version(filename, version)` | Nombre de chunks pour une version donnée. |
| `count()` | Nombre total de chunks. |
| `get_source_files()` | Ensemble des noms de fichiers indexés (sert au filtre « cherche dans X.pdf »). |
| `get_all_documents()` | Tout le contenu (debug/export). |
| `delete_by_source(source_filename)` | Supprime tous les chunks d'un fichier. |
| `list_sources()` | Liste des sources avec nombre de chunks. |
| `clear()` | Vide entièrement la collection. |

---

## 7. COUCHE RETRIEVAL (Module 4)

### `app/retrieval/retriever.py` — Recherche hybride + reranking

**Fonctions module :**

| Fonction | Rôle |
|---|---|
| `_get_cross_encoder()` | Charge le reranker CrossEncoder (ms-marco-MiniLM) depuis le snapshot LOCAL (hors-ligne). Retourne None si indisponible → rerank désactivé proprement. |
| `_tokenize(text)` | Tokenisation simple (minuscule, split) pour BM25. |
| `_save_bm25_cache(index, docs, count)` / `_load_bm25_cache(expected_count)` | Cache disque de l'index BM25 (évite de le reconstruire à chaque question). |
| `_build_bm25_index(vector_store)` | Construit l'index BM25 sur tous les chunks. |
| `_ensure_bm25_index(vector_store)` | Charge le cache si valide, sinon reconstruit. |
| `_rrf_score(rank, k=60)` | Score Reciprocal Rank Fusion : 1/(k + rank). |

**Classe `Retriever` :**

| Méthode | Rôle |
|---|---|
| `__init__(vector_store, llm=None)` | Le LLM optionnel sert au HyDE (requête hypothétique). |
| `retrieve(question, top_k, min_score, where)` | Pipeline complet : recherche vectorielle (top_k×2+2 candidats) → recherche BM25 → fusion RRF → rerank CrossEncoder → top_k finaux. Le paramètre `where` restreint au fichier nommé. |
| `_bm25_search(query, top_k)` | Recherche lexicale BM25 (attrape acronymes et termes rares). |
| `_rrf_merge(vector_results, bm25_results, top_k)` | Fusionne les deux listes par score RRF. |
| `format_context(results)` | Formate les chunks retenus en contexte texte numéroté pour le prompt du LLM. |

---

## 8. COUCHE GÉNÉRATION (Module 5)

### `app/generation/llm.py` — Client Ollama + circuit breaker (M2)

**Classe `OllamaLLM` :**

| Méthode | Rôle |
|---|---|
| `__init__(model_name, base_url, temperature)` | Configure le client HTTP vers Ollama. |
| `health_check(timeout)` | Ping `/api/tags`. Appelé avant chaque génération. |
| `_register_failure()` | Compte un échec consécutif. |
| `circuit_open()` | **Circuit breaker** : True après 3 échecs consécutifs (ouverture 30 s). Pendant l'ouverture, l'UI répond gracieusement sans timeout réseau. |
| `build_prompt(question, context, history)` | Construit le prompt RAG : instructions anti-hallucination + contexte numéroté + question (+ historique de conversation). |
| `generate_stream(question, context, history)` | Génération en streaming (tokens affichés au fur et à mesure). |
| `generate(question, context)` | Génération complète (utilisée par l'évaluation). |
| `parse_structured_response(raw_text)` | Extrait une réponse structurée si le modèle en renvoie une. |
| `generate_hypothetical_answer(question)` | Génère une réponse hypothétique (HyDE) pour améliorer le retrieval. Optionnel, désactivé par défaut. |

---

## 9. CŒUR APPLICATIF

### `app/core/rag_engine.py` — Orchestrateur RAG

**Fonctions module :**

| Fonction | Rôle |
|---|---|
| `_resolve_ollama_base_url()` | URL d'Ollama selon l'environnement (localhost hors Docker, host.docker.internal dedans). |
| `_extract_document_filter(question, vector_store)` | Détecte un nom de fichier dans la question (« cherche dans CV_x.pdf ») et le résout contre l'index → retourne `{"source_file": ...}` ou None. Comparaison insensible à la casse. |
| `_is_followup(question)` | True si question de suivi : très courte (≤ 3 mots) ou commence par un connecteur (« and », « et », « ensuite »…). |
| `build_search_query(question, history)` | **Suivi conversationnel** : réécrit « and now ? » en requête autonome adossée à la dernière question utilisateur. Les questions autonomes ne sont pas modifiées. |

**Constantes :** `TOP_K=6`, `MIN_SCORE=0.0` (seuil neutre : le tri repose sur top_k + rerank),
regex de détection de fichier `_FILENAME_RE`.

**Classe `RAGEngine` :**

| Méthode | Rôle |
|---|---|
| `__init__(chroma_persist_dir, collection_name, model_name, top_k, min_score)` | Assemble VectorStore + Retriever + LLM. Vérifie Ollama (llm=None si absent). Collection unique partagée (décision documentée §6.3). |
| `_check_ollama()` | Disponibilité d'Ollama au démarrage. |
| `ingest_sql(db_path)` | Ingestion d'une base SQLite complète. |
| `ingest_document(file_path, filename, file_data)` | Ingestion d'un document avec versionnage global : hash → si inchangé, no-op ; sinon parse → chunk → record_ingestion (version++) → tag des chunks avec la version → indexation. Retourne (chunks, version). |
| `answer(question)` | Pipeline question → réponse : filtre fichier éventuel → retrieval (hybride + rerank) → formatage du contexte → génération Ollama. Retourne (réponse, contextes structurés pour Ragas). |

---

## 10. COUCHE ÉVALUATION & OBSERVABILITÉ

### `app/evaluation/eval_pipeline.py` — Pipeline Ragas (Sprint 7)

**Constantes :** `EVALUATOR_MODEL="mistral"` (juge), `REPORT_PATH=data/evaluation_report.csv`,
`QUERY_TIMEOUT_SECONDS` (budget/question), `RAGAS_TIMEOUT_SECONDS` (budget métriques),
`EVALUATION_RETRIEVAL_ONLY` (mode rapide sans métriques).

**`GOLD_DATASET`** : 30 questions/réponses de référence couvrant le CDC (stage_si_RAG.pdf),
la base SQL de test, le Markdown et le TXT de test.

| Fonction | Rôle |
|---|---|
| `_run_with_timeout(fn, timeout, default)` | Exécute fn dans un thread avec budget strict (NFR-04). |
| `setup_ragas_evaluator()` | Configure Ragas avec Ollama comme juge (4 métriques : faithfulness, answer_relevancy, context_precision, context_recall). Tolère les imports cassés (langchain v1) → mode retrieval-only. `EVALUATION_RETRIEVAL_ONLY=1` force ce mode. |
| `class EvalResult` | Résultat d'un cas de test : question, reference, response, contexts, 4 métriques, error, latency_ms. Propriété `metrics_available`. |
| `run_evaluation(progress_callback, timeout_seconds)` | Boucle complète : pour chaque question → answer() avec budget → log observabilité (M4) → scoring Ragas avec budget séparé. Retourne la liste des EvalResult. |
| `export_results(results, output_path)` | Export CSV : question, reference, response, 4 métriques, latency_ms, error, nb_contexts, top1_score, top1_source. C'est CE fichier que lit la page Evaluation & Audit. |
| `print_summary(results)` | Résumé console : moyennes vs cibles, erreurs, stats retrieval. |
| `main()` | Point d'entrée CLI : vérifie ChromaDB → run_evaluation → export → summary. |

### `app/observability.py` — Journal des requêtes (M4)

**Classe `QueryLogger` :**

| Méthode | Rôle |
|---|---|
| `__init__(db_path)` | Ouvre query_logs.db. |
| `_connect()` / `_init_db()` | Table query_logs (timestamp, user, question, scores, latences, modèle, sources). |
| `log(question, top1_score, mean_score, num_chunks, llm_latency_ms, total_latency_ms, model_used, source_files)` | Journalise UNE requête (chat ET évaluation → même table). |
| `get_recent(limit)` | Dernières requêtes. |
| `get_stats(limit)` | Agrégats : latence moyenne, score moyen, volume. |

### `app/pages/evaluation.py` — Page Evaluation & Audit (admin)

| Fonction | Rôle |
|---|---|
| `_run_evaluation_background(progress_placeholder, progress_bar)` | Lance run_evaluation dans un thread (UI non bloquée), met à jour la barre de progression, exporte le rapport. |
| `render_evaluation_page()` | Affiche les 4 cartes de métriques (**valeurs entre 0 et 1**) + tableau détaillé du rapport CSV + bouton « Run local evaluation ». Protégé par require_admin(). |

### `app/pages/performance.py` — Page Performance (admin)

| Fonction | Rôle |
|---|---|
| `render_performance_page()` | Lit query_logs.db : latence moyenne, volume de requêtes, scores moyens, graphiques. |

---

## 11. PAGES MÉTIER RESTANTES

### `app/pages/chat.py` — Chat RAG (tous rôles)

| Fonction | Rôle |
|---|---|
| `_render_source_drawer(sources_list)` | Panneau déroulant des sources citées (fichier, page, score, extrait). |
| `_render_citations(citations_list)` | Affiche les citations [1], [2]… dans la réponse. |
| `render_chat_page()` | Page principale : historique persistant, garde salutation (is_greeting), circuit breaker (réponse gracieuse si Ollama down), **réécriture des questions de suivi** (build_search_query), **filtre par fichier** (_extract_document_filter), retrieval → streaming → dédup sources → export PDF. |

### `app/pages/file_indexation.py` — Indexation (admin)

| Fonction | Rôle |
|---|---|
| `_file_type_label(source_file)` | Badge du type de fichier (PDF/TXT/MD/CSV/SQL). |
| `render_file_indexation_page()` | Upload multiple avec progression, ingestion via ingest_uploaded_file, suppression/purge. |

### `app/pages/indexed_files.py` — Consultation de l'index

| Fonction | Rôle |
|---|---|
| `_file_type_label(source_file)` | Badge du type de fichier. |
| `render_indexed_files_page()` | Tableau des fichiers indexés (nom, version, chunks, date). Accessible aux users (lecture seule). |

### `app/pages/system_parameters.py` — Paramètres (admin)

| Fonction | Rôle |
|---|---|
| `render_system_parameters_page()` | Réglages : température LLM, top_k, taille de chunk. Sauvegardés en session. |

---

## 12. TESTS (`tests/`)

| Fichier | Contenu |
|---|---|
| `conftest.py` | Fixtures partagées (chemins, bases temporaires). |
| `test_parser_pytest.py` | Tests du parseur (PDF, TXT, MD, SQL, OCR fallback). |
| `test_chunker_pytest.py` | Tests du chunking adaptatif (stratégies, overlap, tokens). |
| `test_vector_store_pytest.py` | Tests ChromaDB (add/search/delete/métadonnées assainies). |
| `test_retriever_pytest.py` | Tests du retriever (RRF, BM25, format_context). |
| `test_coverage_core.py` | Couverture des modules cœur (Mo5 ≥ 80 %) : OllamaLLM (build_prompt, HyDE), Database (CRUD), Retriever. |
| `test_review_fixes.py` | Verrouille les correctifs de review : C1/Mo4 (versionnage global + migration), C4 (search_latest par égalité), M1 (require_admin), M2 (circuit breaker), C5 (périmètre watcher). |
| `test_retrieval_quality.py` | Qualité RAG : extraction de fichier → where, MIN_SCORE neutre, suivi conversationnel (build_search_query), reranker hors-ligne. |
| `test_integration_embeddings.py` | Intégration embeddings réels (dimension 384). Marker `integration`. |
| `test_pipeline_complet.py` | Round-trip complet Parser → Chunker → VectorStore → Search. Marker `integration`. |
| `test_rag.py` | End-to-end retrieval + génération Ollama (skip si indisponible). Marker `integration`. |
| `test_evaluation_pipeline.py` | Tests du pipeline d'évaluation : export CSV (schéma, valeurs), boucle réelle sur sous-ensemble (EVALUATION_LIMIT, défaut 30 questions). |
| `test_network_sovereignty.py` | Audit LIVE de la souveraineté (à exécuter DANS le conteneur) : URLs externes bloquées, DNS bloqué, Ollama joignable. 9/9 PASS attendus. |
| `test_entrypoint_network_rules.py` | Validation STATIQUE (hors Docker) de la config réseau : entrypoint.sh, docker-compose.yml. |
| `test_app_ui_state.py` | État de session UI (RBAC, navigation). |

**Commandes :**
```bash
python -m pytest tests/ -q                          # unitaires seuls (105)
python -m pytest tests/ -m integration -q           # intégration (14)
docker exec rag-souverain python tests/test_network_sovereignty.py   # audit live
```

---

## 13. DONNÉES (`data/`)

| Chemin | Rôle |
|---|---|
| `data/chroma/` | Base vectorielle ChromaDB persistante (141 chunks). |
| `data/rag.db` | SQLite : users, doc_versions, conversations, chat_messages. |
| `data/query_logs.db` | Journal des requêtes (observabilité M4). |
| `data/documents/` | Documents sources (monté en lecture seule dans le conteneur). |
| `data/uploads/` | Fichiers uploadés via l'interface. |
| `data/evaluation_report.csv` | Rapport d'évaluation lu par la page Evaluation & Audit. |
| `data/test_database.db` | Base SQL de démonstration (employés, produits). |

---

## 14. DIAGRAMMES UML (`docs/uml/`)

Sources PlantUML + rendus PNG prêts pour le rapport :
`use_case`, `class`, `sequence_login`, `sequence_chat`, `sequence_indexation`,
`sequence_evaluation`, `sequence_export_conversation`, `sequence_user_management`,
`sequence_performance`, `activity_chat`, `activity_ingestion`, `component`,
`deployment`, `state`.

---

## 15. FLUX COMPLETS (résumé)

**Ingestion :** Fichier → `DocumentParser` (natif/OCR/texte/SQL) → `AdaptiveChunker`
(section/fenêtre/tableau/SQL) → `RAGEngine.ingest_document` (hash, version globale)
→ `ChromaVectorStore.add_chunks` (embeddings locaux → ChromaDB).

**Question → Réponse :** Question → salutation ? circuit breaker ouvert ?
→ réécriture de suivi (`build_search_query`) → filtre fichier (`_extract_document_filter`)
→ `Retriever.retrieve` (vectoriel + BM25 → RRF → CrossEncoder) → `format_context`
→ `OllamaLLM.generate_stream` (prompt anti-hallucination) → dédup sources → affichage
→ journalisation (`QueryLogger`) + persistance SQLite.

**Évaluation :** GOLD_DATASET (30 questions) → `run_evaluation` (budget par question)
→ métriques Ragas (juge Mistral local) → `export_results` → `evaluation_report.csv`
→ page Evaluation & Audit.
