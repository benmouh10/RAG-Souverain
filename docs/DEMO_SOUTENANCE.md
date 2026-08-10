# Démo de soutenance — RAG Souverain (10 min)

Runbook chronométré couvrant l'ensemble des points forts face au jury.
Chaque étape indique la commande, le résultat attendu et le point à argumenter.

> Prérequis avant la soutenance : Docker Desktop démarré, Ollama démarré
> (`ollama serve`) avec le modèle `mistral` chargé (`ollama pull mistral`).

---

## 1. Démarrer le système (1 min)

```bash
docker compose up --build -d
docker ps --filter "name=rag-" --format "{{.Names}} | {{.Status}}"
# Attendu : rag-souverain (healthy) + rag-ingestion-watcher
```

**À dire :** image multi-stage (modèle d'embeddings pré-intégré), volume persistant
ChromaDB, accès Ollama limité à `host.docker.internal:11434`, DNS forcé sur `127.0.0.1`,
réseau dédié `rag_network`.

## 2. Login + indexation d'un document (1 min)

1. Ouvrir `http://localhost:8501` → se connecter en `admin` / `admin`.
2. Menu **Files Indexation** → *Upload* `rapport_annuel.pdf`.
3. Vérifier l'incrément de version + apparition dans *Indexed Files*.

**À dire :** versionnage global `doc_versions` (C1), ingestion adaptative
(section/sliding-window/table/SQL), parsing PDF natif + OCR.

## 3. Chat RAG — question + sources (2 min)

1. Menu **Chat RAG** → poser : *« Quel est le CA du groupe ? »*
2. Montrer : réponse générée, **sources citées** (fichier/page/score), dédup
   `file|page|content`.

**À dire :** pipeline hybride BM25 + vectoriel (RRF), rerank CrossEncoder **hors-ligne**
(snapshot résolu localement), embedding multilingue 384 dim en mode hors-ligne.

**Démo « suivi conversationnel » :** après la réponse, poser *« and now ? »* →
la question est réécrite en requête autonome adossée à la précédente
(`build_search_query`) et la recherche retrouve le même document.
**Démo « recherche par fichier » :** poser *« cherche dans CV_BENMOUH_Mohamed.pdf les compétences »* →
filtre `where` (BM25 RRF restreint au fichier), sources toutes issues du CV.

## 4. Circuit breaker — Ollama coupé (1 min)

```bash
ollama stop mistral        # ou couper le process Ollama
# (dans l'UI) reposer une question → réponse « service dégradé » immédiate
```

**Attendu :** pas de timeout réseau — message gracieux + bandeau `st.warning`
(Chat → réponse instantanée sans appeler retriever/LLM).

## 5. Rétablissement (1 min)

```bash
ollama serve
# attendre ~30 s (health check périodique)
# (dans l'UI) reposer la question → requête normale
```

**À dire :** le circuit breaker se referme automatiquement dès que `health_check()` repasse.

## 6. Évaluation RAGAS (2 min)

Menu **Evaluation & Audit** → *Lancer l'évaluation* (30 questions, budget 30 s/question).
Ouvrir le rapport : `data/evaluation_report.csv`.

**Attendu :** métriques de fidélité + pertinence calculées localement (Mistral comme
juge — aucun LLM distant), requêtes journalisées dans `query_logs` (M4).

## 7. Souveraineté réseau — démonstration live (1 min)

```bash
docker exec rag-souverain python -c "import urllib.request; print(urllib.request.urlopen('https://google.com', timeout=3).status)"
# Attendu : URLError — Temporary failure in name resolution   →  aucune donnée ne sort

docker exec rag-souverain python tests/test_network_sovereignty.py
# Attendu : 9/9 PASS — SOUVERAINETÉ CONFIRMÉE
```

**À dire :** règles iptables `OUTPUT DROP` par défaut + allowlist loopback/Ollama,
DNS local forcé → en 2 commandes on **prouve** qu'aucune donnée ne sort (NFR-01),
pas une affirmation de conception.

## 8. Tests (1 min)

```bash
docker exec rag-souverain python -m pytest tests/ -m "not integration" -q
# Attendu : 105 passed

docker exec rag-souverain python -m pytest tests/ -m integration -q
# Attendu : 14 passed
```

**À dire :** 105 unitaires + 14 intégration exécutés **dans** le conteneur isolé —
la souveraineté est testée et démontrée, pas seulement documentée.

---

## Questions anticipées du jury

| Question | Réponse |
|---|---|
| « Comment un utilisateur RH ne voit-il pas les documents Finance ? » | Cloisonnement non implémenté dans cette version : § 6.3 documente la décision (collection unique, flux uni-directionnel) et le chemin d'extension (`department` + filtre `$and`). |
| « Pourquoi une seule collection ChromaDB ? » | Flux d'ingestion central/uni-directionnel, documents partagés (référentiels communs) ; RBAC restreint les *actions* (upload/delete), pas la lecture. Décision tracée au § 6.3. |
| « Vos tests d'intégration sont exclus du CI ? » | Exclus du run par défaut car ils exigent Ollama + modèle réel (`pytest.ini` → marker `integration`) ; activables explicitement, exécutés dans le conteneur en démo. |
| « Comment prouvez-vous que rien ne sort ? » | `tests/test_network_sovereignty.py` : 5 URLs externes bloquées + DNS bloqué + seul Ollama local joignable (9/9 PASS en direct). |
