GOLD_DATASET: list[dict[str, str]] = [
    # ---- Questions sur le CDC PDF (stage_si_RAG.pdf) ----
    {
        "user_input": "Quel est l'objectif principal du projet RAG Souverain ?",
        "reference": "Le projet RAG Souverain vise à concevoir et développer un système RAG (Retrieval-Augmented Generation) capable d'ingérer des documents d'entreprise hétérogènes (PDF, textes, bases SQL), de les indexer dans une base vectorielle locale, et de permettre l'interrogation en langage naturel via un modèle de langage open-source déployé entièrement sur l'infrastructure de l'entreprise, sans aucune transmission de données vers des services externes.",
        "source": "stage_si_RAG.pdf (page 2)"
    },
    {
        "user_input": "Quels sont les quatre objectifs critiques du projet ?",
        "reference": "Les quatre objectifs critiques sont : O1 - Construire un pipeline d'ingestion automatique de documents PDF, TXT et SQL ; O2 - Implémenter un chunker adaptatif selon le type de document ; O3 - Vectoriser les chunks et les stocker dans une base vectorielle locale ; O4 - Intégrer un LLM local (Mistral 7B via Ollama) comme moteur de génération.",
        "source": "stage_si_RAG.pdf (page 6)"
    },
    {
        "user_input": "Quelle est la technologie choisie pour le LLM local et pourquoi ?",
        "reference": "La technologie choisie est Ollama avec Mistral 7B Instruct. Les justifications sont : excellent support du français et de l'anglais, capacité à tourner sur CPU et GPU, et licence Apache 2.0.",
        "source": "stage_si_RAG.pdf (page 8)"
    },
    {
        "user_input": "Quelle est la technologie choisie pour les embeddings ?",
        "reference": "La technologie choisie pour les embeddings est sentence-transformers, car c'est une solution 100% locale avec des modèles multilingues disponibles.",
        "source": "stage_si_RAG.pdf (page 8)"
    },
    {
        "user_input": "Quelle est la technologie choisie pour la base vectorielle ?",
        "reference": "La technologie choisie est ChromaDB en développement (zero-config en Python), puis Qdrant pour la production (Docker-natif).",
        "source": "stage_si_RAG.pdf (page 8)"
    },
    {
        "user_input": "Comment la souveraineté des données est-elle assurée au niveau réseau ?",
        "reference": "La souveraineté est assurée par le réseau Docker défini avec internal: true, ce qui interdit toute connexion sortante vers Internet.",
        "source": "stage_si_RAG.pdf (page 9)"
    },
    {
        "user_input": "Quelles sont les quatre métriques d'évaluation RAGAS et leurs cibles ?",
        "reference": "Les quatre métriques sont : Context Recall (cible ≥ 0.75) qui mesure si les chunks pertinents sont tous retrouvés ; Context Precision (cible ≥ 0.70) qui mesure le bruit dans les chunks récupérés ; Faithfulness (cible ≥ 0.80) qui mesure la fidélité aux sources sans hallucination ; Answer Relevancy (cible ≥ 0.70) qui mesure si la réponse répond à la question.",
        "source": "stage_si_RAG.pdf (page 20)"
    },
    {
        "user_input": "Quels sont les indicateurs de performance cibles du système ?",
        "reference": "Les indicateurs sont : Latence de réponse < 10 secondes, Débit d'indexation ≥ 20 pages/min, Précision retrieval Recall@4 ≥ 0.75, Faithfulness score RAGAS ≥ 0.80, Disponibilité 100%.",
        "source": "stage_si_RAG.pdf (page 16)"
    },
    {
        "user_input": "Quelles sont les fonctionnalités du module de retrieval ?",
        "reference": "Les fonctionnalités sont : F4.1 - Conversion de la question en vecteur ; F4.2 - Recherche par similarité cosinus ; F4.3 - Retour des top-k chunks avec score de similarité (k paramétrable, défaut: 4) ; F4.4 - Filtrage par seuil de score minimum ; F4.5 - Filtrage par métadonnées.",
        "source": "stage_si_RAG.pdf (page 13)"
    },
    {
        "user_input": "Quelles sont les fonctionnalités du module de génération LLM ?",
        "reference": "Les fonctionnalités sont : F5.1 - Envoi du prompt au LLM Mistral 7B via l'API Ollama locale ; F5.2 - Streaming de la réponse ; F5.3 - Injection des sources dans le prompt pour ancrage factuel ; F5.4 - Affichage des sources citées ; F5.5 - Paramétrage de la température.",
        "source": "stage_si_RAG.pdf (page 14)"
    },
    {
        "user_input": "Quels sont les endpoints principaux de l'API REST FastAPI ?",
        "reference": "Les endpoints sont : POST /ingest pour envoyer un fichier pour indexation ; POST /query pour poser une question ; GET /status pour le statut du système ; DELETE /index pour supprimer l'index ; GET /documents pour lister les documents indexés.",
        "source": "stage_si_RAG.pdf (page 15)"
    },
    {
        "user_input": "Quels sont les risques identifiés et leurs mitigations ?",
        "reference": "Les risques sont : Machine insuffisante pour Mistral 7B (mitigation: utiliser phi3:mini comme fallback) ; PDF scannés de mauvaise qualité (mitigation: prétraitement image avant OCR) ; Chunking inadapté (mitigation: tests unitaires dès S3) ; Dérive de scope (mitigation: respect strict du périmètre CDC, MoSCoW) ; Manque de temps pour Docker (mitigation: priorisation).",
        "source": "stage_si_RAG.pdf (page 21)"
    },
    {
        "user_input": "Quels sont les livrables attendus du projet ?",
        "reference": "Les livrables sont : L1 - Code source versionné sur Git avec README complet ; L2 - Docker Compose pour démarrer le système en une commande ; L3 - Rapport technique documentant l'architecture ; L4 - Rapport d'évaluation avec métriques RAGAS ; L5 - Présentation de soutenance ; L6 - Cahier des charges finalisé et signé.",
        "source": "stage_si_RAG.pdf (page 22)"
    },
    {
        "user_input": "Qu'est-ce que le chunking adaptatif et quelles sont ses stratégies ?",
        "reference": "Le chunking adaptatif découpe le texte extrait en segments de taille optimale, avec une stratégie adaptée automatiquement au type de document. Les stratégies sont : PDF structuré (par section, overlap 0), PDF libre/texte (par fenêtre glissante, 512 tokens, overlap 50), Markdown (par section), Tableau (préserve les lignes intactes), SQL (un enregistrement = un chunk).",
        "source": "stage_si_RAG.pdf (page 11)"
    },
    {
        "user_input": "Quel est le découpage en sprints du projet ?",
        "reference": "Le projet est découpé en 8 sprints sur 8 semaines : S1 Cadrage et setup, S2 Ingestion data, S3 Chunking, S4 Vectorisation, S5 RAG et génération, S6 Interface utilisateur et API, S7 Évaluation, S8 Finalisation et déploiement.",
        "source": "stage_si_RAG.pdf (page 18)"
    },
    {
        "user_input": "Qu'est-ce que le système RAG ?",
        "reference": "RAG signifie Retrieval-Augmented Generation. C'est une technique combinant recherche documentaire et génération de texte par LLM.",
        "source": "stage_si_RAG.pdf (page 23)"
    },
    {
        "user_input": "Qu'est-ce qu'un embedding ?",
        "reference": "Un embedding est une représentation vectorielle numérique d'un texte capturant son sens sémantique.",
        "source": "stage_si_RAG.pdf (page 23)"
    },
    {
        "user_input": "Qu'est-ce qu'un chunk ?",
        "reference": "Un chunk est un fragment de texte issu du découpage d'un document, unité de base de l'indexation.",
        "source": "stage_si_RAG.pdf (page 23)"
    },
    {
        "user_input": "Qu'est-ce qu'un Vector Store ?",
        "reference": "Un Vector Store est une base de données spécialisée pour le stockage et la recherche par similarité de vecteurs d'embeddings.",
        "source": "stage_si_RAG.pdf (page 23)"
    },
    {
        "user_input": "Quels sont les deux pipelines du système RAG ?",
        "reference": "Le système est composé de deux pipelines distincts mais interconnectés : le pipeline d'ingestion (offline) pour le traitement des documents et la construction de l'index vectoriel, et le pipeline de requête (online) pour le traitement des questions utilisateur et la génération de réponses.",
        "source": "stage_si_RAG.pdf (page 7)"
    },
    # ---- Questions sur la base SQL (test_database.db) ----
    {
        "user_input": "Quels sont les employés présents dans la base de données ?",
        "reference": "Les employés sont : Jean Dupont (32 ans, Développeur, embauché le 2022-01-15), Sophie Martin (28 ans, Designer, embauchée le 2023-03-20), et Paul Durand (40 ans, Chef de projet, embauché le 2020-06-01).",
        "source": "test_database.db (table employes)"
    },
    {
        "user_input": "Quels sont les produits disponibles dans la base de données ?",
        "reference": "Les produits sont : Ordinateur portable (code P001, 899.99€, stock: 10), Souris sans fil (code P002, 29.99€, stock: 150), et Clavier mécanique (code P003, 129.99€, stock: 50).",
        "source": "test_database.db (table produits)"
    },
    {
        "user_input": "Quel est le prix du clavier mécanique ?",
        "reference": "Le clavier mécanique (code P003) coûte 129.99€ et il y en a 50 en stock.",
        "source": "test_database.db (table produits)"
    },
    {
        "user_input": "Qui est le chef de projet dans la base employés ?",
        "reference": "Paul Durand, 40 ans, est le Chef de projet. Il a été embauché le 2020-06-01.",
        "source": "test_database.db (table employes)"
    },
    {
        "user_input": "Combien de souris sans fil sont en stock ?",
        "reference": "Il y a 150 souris sans fil (code P002) en stock, au prix unitaire de 29.99€.",
        "source": "test_database.db (table produits)"
    },
    # ---- Questions sur le fichier Markdown de test ----
    {
        "user_input": "Que contient le fichier Markdown de test ?",
        "reference": "Le fichier Markdown de test contient une introduction, un tableau avec les noms, âges et villes d'Alice (30 ans, Paris), Bob (25 ans, Lyon) et Charlie (35 ans, Marseille), et une conclusion.",
        "source": "test_markdown.md"
    },
    {
        "user_input": "Quel est l'âge de Bob dans le tableau du fichier Markdown ?",
        "reference": "Bob a 25 ans et habite à Lyon.",
        "source": "test_markdown.md"
    },
    {
        "user_input": "Quelles sont les villes des personnes dans le tableau Markdown ?",
        "reference": "Les villes sont : Paris pour Alice, Lyon pour Bob, et Marseille pour Charlie.",
        "source": "test_markdown.md"
    },
    # ---- Questions sur le fichier texte de test ----
    {
        "user_input": "Que contient le fichier texte de test ?",
        "reference": "Le fichier texte de test contient plusieurs lignes pour vérifier l'ingestion et l'encodage UTF-8, avec des caractères spéciaux français (éèàôûçñ) et des ligatures (œæ).",
        "source": "test_text.txt"
    },
    {
        "user_input": "Quel est le but de la dernière phrase du fichier texte de test ?",
        "reference": "La dernière phrase \"Ceci est un document très simple pour valider l'ingestion de base.\" confirme que le fichier sert de validation basique du pipeline.",
        "source": "test_text.txt"
    },
    # ---- Questions Additionnelles Générales (Documents data/uploads) ----
    {
        "user_input": "Qui est Mohamed Ben Mouh ?",
        "reference": "Mohamed Ben Mouh est mentionné dans le CV fourni dans les documents. Il possède des compétences et une expérience détaillées dans son CV.",
        "source": "Mohamed_Ben_Mouh_CV.pdf"
    },
    {
        "user_input": "What is the primary role mentioned in Mohamed Ben Mouh's CV?",
        "reference": "Le CV détaille les expériences, projets et la formation de Mohamed Ben Mouh, généralement lié à un profil technique, data ou ingénierie.",
        "source": "Mohamed_Ben_Mouh_CV.pdf"
    },
    {
        "user_input": "What are the main topics covered in the clinical handbook?",
        "reference": "Le manuel clinique (clinical handbook) couvre des directives médicales, des procédures de santé, et des standards de soins pour les patients.",
        "source": "clinical_handbook.pdf"
    },
    {
        "user_input": "Quelles sont les recommandations générales dans le clinical handbook ?",
        "reference": "Il fournit des recommandations sur le traitement des patients, les protocoles cliniques et les bonnes pratiques médicales.",
        "source": "clinical_handbook.pdf"
    },
    {
        "user_input": "Que contient la base de données clinical_rag.db ?",
        "reference": "Elle contient des données de patients, incluant des informations comme les taux d'hémoglobine, les niveaux de ferritine, et d'autres données médicales.",
        "source": "clinical_rag.db"
    },
    {
        "user_input": "Quelles informations trouve-t-on sur Sara Bennani dans clinical_rag.db ?",
        "reference": "On y trouve des données cliniques la concernant, comme son taux d'hémoglobine ou de ferritine.",
        "source": "clinical_rag.db"
    },
    {
        "user_input": "Quelles informations contient le document security_architecture.pdf ?",
        "reference": "Il décrit l'architecture de sécurité, les protocoles de protection des données, les réseaux isolés et les mesures de mitigation des risques.",
        "source": "security_architecture.pdf"
    },
    {
        "user_input": "Comment la sécurité est-elle structurée selon le document d'architecture de sécurité ?",
        "reference": "Elle est structurée autour de pare-feux, de réseaux isolés (comme Docker internal), du chiffrement et de contrôles d'accès stricts.",
        "source": "security_architecture.pdf"
    },
    {
        "user_input": "Quel est le sujet du fichier mails-RH.pdf ?",
        "reference": "Ce document compile des correspondances et des emails relatifs aux ressources humaines (RH), incluant des politiques internes, des annonces ou des communications aux employés.",
        "source": "mails-RH.pdf"
    },
    {
        "user_input": "Quelles annonces RH sont présentes dans les mails RH ?",
        "reference": "Les emails traitent des congés, des intégrations de nouveaux employés, des formations ou d'autres communications internes RH.",
        "source": "mails-RH.pdf"
    },
    {
        "user_input": "Que contient le document SQL Complete Notes .pdf ?",
        "reference": "C'est un manuel complet sur le langage SQL, couvrant les requêtes de base, les jointures, les sous-requêtes, l'optimisation et la gestion des bases de données.",
        "source": "SQL Complete Notes .pdf"
    },
    {
        "user_input": "What are the fundamental SQL commands explained in the SQL notes?",
        "reference": "Il explique des commandes fondamentales comme SELECT, INSERT, UPDATE, DELETE, CREATE, DROP et ALTER.",
        "source": "SQL Complete Notes .pdf"
    },
    {
        "user_input": "Quel est le contenu du fichier README.md ?",
        "reference": "Le fichier README.md contient les instructions principales du projet, comment l'installer, le configurer, et comment démarrer le système (par exemple via Docker Compose).",
        "source": "README.md"
    },
    {
        "user_input": "Que trouve-t-on dans le fichier github-recovery-codes.txt ?",
        "reference": "Ce fichier contient des codes de secours (recovery codes) pour l'accès sécurisé à un compte GitHub, utilisés pour l'authentification à deux facteurs.",
        "source": "github-recovery-codes.txt"
    },
    {
        "user_input": "À quoi sert le document rag_evaluation_guide.pdf ?",
        "reference": "Il sert de guide pour évaluer le système RAG, expliquant les métriques comme Faithfulness, Answer Relevancy, Context Precision et Context Recall.",
        "source": "rag_evaluation_guide.pdf"
    },
    {
        "user_input": "Que contient la base de données project_rag.db ?",
        "reference": "Elle contient les informations relatives à la gestion de projets, aux jalons, aux équipes assignées et aux statuts d'avancement.",
        "source": "project_rag.db"
    },
    {
        "user_input": "Comment le RAG peut-il interagir avec project_rag.db ?",
        "reference": "Le système RAG utilise des requêtes SQL structurées pour extraire directement les informations sur les projets depuis cette base SQLite.",
        "source": "project_rag.db"
    },
    {
        "user_input": "What is covered in the sql-language-quick-reference.pdf?",
        "reference": "C'est un guide de référence rapide pour les syntaxes SQL courantes, incluant les fonctions d'agrégation, les clauses WHERE et GROUP BY.",
        "source": "sql-language-quick-reference.pdf"
    },
    {
        "user_input": "Que contient le fichier 'more emails.txt' ?",
        "reference": "Il contient une collection supplémentaire d'emails, souvent utilisée pour tester l'ingestion de données textuelles non structurées.",
        "source": "more emails.txt"
    },
    {
        "user_input": "Que contient le fichier 'pour opencode.txt' ?",
        "reference": "C'est un fichier texte contenant des notes, du code, ou des instructions spécifiques liées à l'initiative 'opencode'.",
        "source": "pour opencode.txt"
    },
    {
        "user_input": "Qu'est-ce que le fichier 'الباقيات الصالحات.pdf' aborde ?",
        "reference": "Ce document en arabe traite de sujets religieux ou de textes classiques, utilisé pour tester la capacité du RAG à traiter des documents multilingues (arabe).",
        "source": "الباقيات الصالحات.pdf"
    },
    {
        "user_input": "What is the purpose of 'الباقيات الصالحات لتثبيت نهايات الآيات.pdf'?",
        "reference": "Il aide à la mémorisation et à la révision des fins de versets (آيات) du Coran, testant l'indexation de textes en langue arabe.",
        "source": "الباقيات الصالحات لتثبيت نهايات الآيات.pdf"
    },
    {
        "user_input": "Quelles sont les différences de gestion entre les PDFs structurés et libres ?",
        "reference": "Les PDFs structurés sont découpés par section (overlap 0) pour préserver la logique, tandis que les PDFs libres utilisent une fenêtre glissante (ex: 512 tokens, overlap 50).",
        "source": "stage_si_RAG.pdf"
    },
    {
        "user_input": "Que signifie RAG dans le contexte de ce projet ?",
        "reference": "Retrieval-Augmented Generation, combinant une recherche de documents locaux avec un LLM open-source (Mistral).",
        "source": "stage_si_RAG.pdf"
    },
    {
        "user_input": "Comment le RAG assure-t-il qu'il n'y a pas d'hallucination ?",
        "reference": "Il est instruit de répondre uniquement à partir des extraits fournis et de dire explicitement si l'information n'est pas disponible.",
        "source": "stage_si_RAG.pdf"
    },
    {
        "user_input": "Quel est le format des données envoyées par le LLM du RAG ?",
        "reference": "Le LLM peut générer du texte en streaming ou structurer sa réponse en JSON avec les citations extraites des documents.",
        "source": "stage_si_RAG.pdf"
    },
    {
        "user_input": "Quels types de documents sont supportés par le RAG Souverain ?",
        "reference": "Le système supporte les PDF, les fichiers textes (.txt, .md, .csv) et les bases de données SQLite (.db).",
        "source": "stage_si_RAG.pdf"
    },
    {
        "user_input": "Quelle métrique RAGAS pénalise un système qui invente des faits ?",
        "reference": "La métrique Faithfulness (fidélité) pénalise fortement un modèle qui génère des faits non présents dans les documents récupérés.",
        "source": "rag_evaluation_guide.pdf"
    },
    {
        "user_input": "Quelle métrique RAGAS évalue si la réponse répond bien à la question ?",
        "reference": "La métrique Answer Relevancy évalue la pertinence de la réponse par rapport à l'intention de la question originale.",
        "source": "rag_evaluation_guide.pdf"
    },
    {
        "user_input": "Comment fonctionne la recherche hybride (ChromaDB + SQLite) ?",
        "reference": "Le système RAG analyse la question : s'il détecte une requête structurée, il génère du SQL et interroge SQLite ; sinon, il effectue une recherche sémantique classique dans ChromaDB.",
        "source": "stage_si_RAG.pdf"
    }
]
