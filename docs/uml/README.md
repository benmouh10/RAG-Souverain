# Diagrammes UML — RAG Souverain

Sources **PlantUML** pour l'étude de conception (voir `CONCEPTION_STUDY.md`).

## Contenu

| Fichier | Diagramme |
|---|---|
| `use_case.puml` | Diagramme de cas d'utilisation (acteurs Admin / Utilisateur, RBAC) |
| `class.puml` | Diagramme de classes (ingestion, indexation, retrieval, génération, évaluation, persistance) |
| `sequence_login.puml` | Séquence — connexion et authentification RBAC |
| `sequence_chat.puml` | Séquence — pipeline question → réponse (chat RAG) + bridge session_state↔SQLite (C3) + export PDF |
| `sequence_performance.puml` | Séquence — page Performance : relecture SQLite au chargement (C3) |
| `sequence_export_conversation.puml` | Séquence — export de la conversation en PDF (UC_Export, Mo2) |
| `sequence_indexation.puml` | Séquence — indexation d'un fichier avec versionnage global |
| `sequence_user_management.puml` | Séquence — gestion des utilisateurs (admin, garde M1) |
| `sequence_evaluation.puml` | Séquence — évaluation Ragas (timeout 30 s / question) |
| `activity_chat.puml` | Activité — pipeline RAG (salutation, retrieval hybride, génération, health check) |
| `activity_ingestion.puml` | Activité — ingestion et indexation de documents |
| `component.puml` | Composants — UI, pipeline RAG, persistance, rapport CSV |
| `deployment.puml` | Déploiement — Docker Compose, réseau isolé, Ollama hôte, volumes |
| `state.puml` | États — session RBAC, versionnage des fichiers et santé LLM |

## Rendu

Les rendus **PNG** sont disponibles dans `docs/uml/png/` (prêts pour le rapport de stage).

Outil en ligne : https://www.plantuml.com/plantuml (copier le contenu d'un fichier).

En local (Java requis) :

```bash
plantuml -tsvg docs/uml/use_case.puml
plantuml -tpng docs/uml/*.puml
```

Extension VS Code : *PlantUML* (jebbs.plantuml) — aperçu en direct.
