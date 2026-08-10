"""Tests de couverture des modules cœur (objectif Mo5 : ≥ 80 %).

Ciblent les fonctions sans dépendance réseau/lourde :
  - OllamaLLM : build_prompt, generate, parse_structured_response, HyDE ;
  - Database  : CRUD utilisateurs, conversations, historique de chat ;
  - Retriever : tokenisation, RRF, format_context, _bm25_search, retrieve (mock VS).
"""

from __future__ import annotations

from app.indexing.vector_store import SearchResult


def _search_result(doc_id: str, content: str, score: float, source: str = "src.txt", page: int = 1) -> SearchResult:
    return SearchResult(
        id=doc_id,
        content=content,
        score=score,
        metadata={"source_file": source, "page_number": page},
    )


class TestOllamaLLMHelpers:
    """Fonctions pures du client LLM (aucun appel réseau)."""

    def test_build_prompt_shape(self):
        from app.generation.llm import OllamaLLM

        llm = OllamaLLM(model_name="mistral", temperature=0.1)
        payload = llm.build_prompt("ma question", "contexte fourni")

        assert payload["model"] == "mistral"
        assert payload["stream"] is True
        assert payload["options"]["temperature"] == 0.1
        assert len(payload["messages"]) == 2
        assert "contexte fourni" in payload["messages"][0]["content"]
        assert "ma question" in payload["messages"][1]["content"]

    def test_build_prompt_with_history_window(self):
        from app.generation.llm import OllamaLLM

        llm = OllamaLLM()
        history = [
            {"role": "user", "content": "ancien"},
            {"role": "assistant", "content": "réponse"},
        ]
        payload = llm.build_prompt("q", "ctx", conversation_history=history)

        user_content = payload["messages"][1]["content"]
        assert "HISTORIQUE DE CONVERSATION" in user_content
        assert "ancien" in user_content and "réponse" in user_content

    def test_generate_aggregates_stream(self):
        from app.generation.llm import OllamaLLM

        llm = OllamaLLM()
        llm.generate_stream = lambda *a, **k: iter(["partie ", "une", " réponse"])
        assert llm.generate("q", "ctx") == "partie une réponse"

    def test_parse_structured_response_plain_json(self):
        from app.generation.llm import OllamaLLM

        raw = '{"answer": "réponse", "citations": [{"chunk_id": "c1", "quote": "extrait"}]}'
        parsed = OllamaLLM.parse_structured_response(raw)

        assert parsed["answer"] == "réponse"
        assert len(parsed["citations"]) == 1

    def test_parse_structured_response_markdown_wrapped(self):
        from app.generation.llm import OllamaLLM

        raw = '```json\n{"answer": "réponse", "citations": []}\n```'
        parsed = OllamaLLM.parse_structured_response(raw)

        assert parsed["answer"] == "réponse"
        assert parsed["citations"] == []

    def test_parse_structured_response_fallback(self):
        from app.generation.llm import OllamaLLM

        parsed = OllamaLLM.parse_structured_response("pas du json")

        assert parsed["answer"] == "pas du json"
        assert parsed["citations"] == []

    def test_parse_structured_response_missing_answer_key(self):
        from app.generation.llm import OllamaLLM

        parsed = OllamaLLM.parse_structured_response('{"foo": "bar"}')

        assert parsed["answer"] == '{"foo": "bar"}'
        assert parsed["citations"] == []

    def test_generate_hypothetical_answer_success(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"message": {"content": "hypothèse générée"}}

        monkeypatch.setattr("requests.post", lambda url, json, timeout: _Resp())
        llm = OllamaLLM(base_url="http://localhost:1")

        assert llm.generate_hypothetical_answer("question") == "hypothèse générée"

    def test_generate_hypothetical_answer_fallback(self, monkeypatch):
        from app.generation.llm import OllamaLLM

        def _boom(url, json, timeout):
            raise ConnectionError("refused")

        monkeypatch.setattr("requests.post", _boom)
        llm = OllamaLLM(base_url="http://localhost:1")

        assert llm.generate_hypothetical_answer("question") == "question"


class TestDatabaseUserAndChat:
    """CRUD utilisateurs, conversations et historique (app/database.py)."""

    def test_user_crud_and_authentication(self, tmp_path):
        from app.database import Database

        db = Database(tmp_path / "rag.db")
        user = db.create_user("alice", "secret", role="admin")

        assert user is not None and user.username == "alice" and user.role == "admin"
        assert db.create_user("alice", "autre") is None  # doublon refusé
        assert db.authenticate("alice", "secret") is not None
        assert db.authenticate("alice", "mauvais") is None
        assert db.authenticate("inconnu", "x") is None
        assert db.get_user_by_username("alice").user_id == user.user_id
        assert db.get_user_by_id(user.user_id).username == "alice"
        assert {u.username for u in db.list_users()} >= {"admin", "alice"}

        assert db.change_password(user.user_id, "nouveau") is True
        assert db.authenticate("alice", "nouveau") is not None
        assert db.authenticate("alice", "secret") is None

        assert db.delete_user(user.user_id) is True
        assert db.get_user_by_id(user.user_id) is None

    def test_conversations_lifecycle(self, tmp_path):
        from app.database import Database

        db = Database(tmp_path / "rag.db")
        user = db.create_user("bob", "pass")

        conv_id = db.create_conversation(user.user_id, title="Première")
        assert conv_id > 0
        assert db.list_conversations(user.user_id)[0]["title"] == "Première"

        assert db.rename_conversation(conv_id, "Renommée") is True
        assert db.list_conversations(user.user_id)[0]["title"] == "Renommée"

        db.save_chat_message(user.user_id, "user", "bonjour", conversation_id=conv_id)
        db.save_chat_message(user.user_id, "assistant", "salut", conversation_id=conv_id)

        history = db.get_chat_history(user.user_id, conversation_id=conv_id)
        assert [h["role"] for h in history] == ["user", "assistant"]

        assert db.delete_conversation(conv_id) is True
        assert db.get_chat_history(user.user_id, conversation_id=conv_id) == []

    def test_chat_history_sources_and_cleanup(self, tmp_path):
        from app.database import Database

        db = Database(tmp_path / "rag.db")
        user = db.create_user("carol", "pass")

        db.save_chat_message(
            user.user_id, "assistant", "réponse",
            sources=[{"content": "extrait", "score": 0.9, "source_file": "a.txt"}],
        )
        db.save_chat_message(user.user_id, "user", "None")  # contenu non pertinent
        db.save_chat_message(user.user_id, "user", "question réelle")

        history = db.get_chat_history(user.user_id)
        assert len(history) == 3
        assert history[0]["sources"][0]["source_file"] == "a.txt"

        # Insertion directe d'un message ancien pour tester cleanup_old_messages
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO chat_history (user_id, role, content, created_at) VALUES (?, 'user', 'vieux', 0.0)",
                (user.user_id,),
            )
        assert db.cleanup_old_messages(max_age_days=1) >= 1

        # clean des messages 'None'
        assert db.cleanup_none_messages(user.user_id) >= 0
        remaining = db.get_chat_history(user.user_id)
        assert all(h["content"] for h in remaining)

        db.clear_chat_history(user.user_id)
        assert db.get_chat_history(user.user_id) == []


class TestRetrieverPure:
    """Fonctions du retriever sans ChromaDB ni modèle (app/retrieval/retriever.py)."""

    def test_tokenize(self):
        from app.retrieval.retriever import _tokenize

        assert _tokenize("Le RAG Souverain, FR!") == ["le", "rag", "souverain", "fr"]

    def test_rrf_score(self):
        from app.retrieval.retriever import _rrf_score

        assert _rrf_score(1) > _rrf_score(2)
        assert 0 < _rrf_score(1) < 1

    def test_rrf_merge_dedupes_and_normalizes(self):
        from app.retrieval.retriever import Retriever

        vector = [
            _search_result("a", "contenu a", 0.9),
            _search_result("b", "contenu b", 0.8),
            _search_result("c", "contenu c", 0.7),
        ]
        bm25 = [
            _search_result("c", "contenu c", 0.6),
            _search_result("d", "contenu d", 0.5),
        ]

        merged = Retriever._rrf_merge(vector, bm25, top_k=3)

        assert len(merged) == 3
        assert {r.id for r in merged} <= {"a", "b", "c", "d"}
        assert all(0.0 <= r.score <= 1.0 for r in merged)

    def test_format_context(self):
        from app.retrieval.retriever import Retriever

        retriever = Retriever(object())
        assert retriever.format_context([]) == ""

        ctx = retriever.format_context([
            _search_result("a", "  contenu A  ", 0.9, source="doc.pdf", page=3),
            _search_result("b", "contenu B", 0.8, source="notes.txt", page=-1),
        ])

        assert "[Extrait 1 - doc.pdf - page 3]" in ctx
        assert "[Extrait 2 - notes.txt]" in ctx
        assert "contenu A" in ctx and "contenu B" in ctx

    def test_retrieve_hybrid_off_with_min_score(self, monkeypatch):
        from app.retrieval import retriever as r

        class _FakeVS:
            def search(self, query, top_k, where=None):
                return [
                    _search_result("a", "contenu a", 0.9),
                    _search_result("b", "contenu b", 0.4),
                ]

        monkeypatch.setattr(r, "USE_HYBRID", False)
        monkeypatch.setattr(r, "USE_HYDE", False)
        monkeypatch.setattr(r, "_get_cross_encoder", lambda: None)

        retriever = r.Retriever(_FakeVS())
        results = retriever.retrieve("question", top_k=6, min_score=0.65)

        assert [res.id for res in results] == ["a"]

    def test_retrieve_returns_empty_when_below_threshold(self, monkeypatch):
        from app.retrieval import retriever as r

        class _FakeVS:
            def search(self, query, top_k, where=None):
                return [_search_result("a", "contenu a", 0.1)]

        monkeypatch.setattr(r, "USE_HYBRID", False)
        monkeypatch.setattr(r, "USE_HYDE", False)

        retriever = r.Retriever(_FakeVS())
        assert retriever.retrieve("question", top_k=6, min_score=0.65) == []

    def test_retrieve_hyde_expansion_and_fallback(self, monkeypatch):
        from app.retrieval import retriever as r

        calls = {"queries": []}

        class _FakeVS:
            def search(self, query, top_k, where=None):
                calls["queries"].append(query)
                return [_search_result("a", "contenu a", 0.9)]

        class _FakeLLM:
            def generate_hypothetical_answer(self, question):
                raise ConnectionError("llm down")

        monkeypatch.setattr(r, "USE_HYBRID", False)
        monkeypatch.setattr(r, "USE_HYDE", True)
        monkeypatch.setattr(r, "_get_cross_encoder", lambda: None)

        retriever = r.Retriever(_FakeVS(), llm=_FakeLLM())
        results = retriever.retrieve("question", top_k=6, min_score=0.1)
        assert results
        assert calls["queries"] == ["question"]  # repli sur la requête brute

    def test_bm25_search(self, monkeypatch):
        from app.retrieval import retriever as r

        class _FakeBM25:
            def get_scores(self, tokens):
                return [5.0, 0.0]

        docs = [
            {"id": "1", "text": "le chat dort", "metadata": {"source_file": "a.txt", "page_number": 1}},
            {"id": "2", "text": "le chien court", "metadata": {"source_file": "b.txt", "page_number": 1}},
        ]

        class _FakeVS:
            class _Collection:
                def get(self, ids=None, include=None):
                    wanted = {"1"}
                    return {
                        "ids": [i for i in ids if i in wanted],
                        "documents": ["le chat dort"],
                        "metadatas": [{"source_file": "a.txt", "page_number": 1}],
                    }

            collection = _Collection()

        monkeypatch.setattr(r, "_bm25_index", _FakeBM25())
        monkeypatch.setattr(r, "_bm25_docs", docs)

        results = r.Retriever(_FakeVS())._bm25_search("chat", top_k=5)

        assert len(results) == 1
        assert results[0].id == "1"
        assert results[0].score == 1.0  # normalisé sur le max des scores BM25

    def test_bm25_search_empty_index(self, monkeypatch):
        from app.retrieval import retriever as r

        monkeypatch.setattr(r, "_bm25_index", None)
        monkeypatch.setattr(r, "_bm25_docs", [])

        assert r.Retriever(object())._bm25_search("question", top_k=5) == []

    def test_ensure_bm25_index_rebuilds_on_count_change(self, monkeypatch):
        from app.retrieval import retriever as r

        built = {"n": 0}

        class _FakeVS:
            def count(self):
                return 3

            def get_all_documents(self):
                built["n"] += 1
                return []

        monkeypatch.setattr(r, "_build_bm25_index", lambda vs: built.__setitem__("n", built["n"] + 1))
        monkeypatch.setattr(r, "_bm25_index", None)
        monkeypatch.setattr(r, "_bm25_collection_count", 0)

        r._ensure_bm25_index(_FakeVS())
        assert built["n"] == 1


class TestVectorStoreCoverage:
    """Fonctions de ChromaVectorStore testables sans charger de modèle."""

    def _add(self, vs, filename: str, content: str, version: int = 1):
        from app.ingestion.chunker import ChunkStrategy, TextChunk

        vs.add_chunks(
            [
                TextChunk(
                    content=content,
                    source_file=filename,
                    page_number=1,
                    chunk_index=0,
                    strategy=ChunkStrategy.SLIDING_WINDOW,
                    version=version,
                )
            ]
        )

    def test_list_sources_and_get_all_documents(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "a.txt", "Premier contenu.")
        self._add(vs, "a.txt", "Second contenu.")
        self._add(vs, "b.txt", "Troisième contenu.")

        sources = vs.list_sources()
        by_file = {s["source_file"]: s["chunk_count"] for s in sources}
        assert by_file["a.txt"] == 2 and by_file["b.txt"] == 1

        docs = vs.get_all_documents()
        assert len(docs) == 3
        assert all("source_file" in d["metadata"] for d in docs)
        assert all("page_number" in d["metadata"] for d in docs)

    def test_search_with_where_filter(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "a.txt", "contenu cible A")
        self._add(vs, "b.txt", "contenu cible B")

        results = vs.search("contenu", top_k=5, where={"source_file": "a.txt"})

        assert results
        assert all(r.metadata["source_file"] == "a.txt" for r in results)

    def test_get_latest_version_per_file(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        self._add(vs, "a.txt", "première version.", version=1)
        self._add(vs, "a.txt", "deuxième version.", version=2)
        self._add(vs, "b.txt", "unique.", version=1)

        assert vs.get_latest_version("a.txt") == 2
        assert vs.get_latest_version("b.txt") == 1
        assert vs.get_latest_version("absente.txt") == 0
        assert vs.count_by_version("a.txt", 1) == 1
        assert vs.count_by_version("a.txt", 2) == 1
        assert vs.get_latest_version_all() == 2

    def test_add_chunks_empty(self, vector_store_fixture):
        vs = vector_store_fixture
        vs.initialize_collection()
        assert vs.add_chunks([]) == 0

    def test_safe_meta_defaults(self):
        from app.indexing.vector_store import _safe_meta

        safe = _safe_meta({"source_file": "None", "page_number": None, "version": None})
        assert safe["source_file"] == "unknown"
        assert safe["page_number"] == 0
        assert safe["version"] == 1
        assert _safe_meta(None)["source_file"] == "unknown"

    def test_resolve_local_sentence_transformers_path(self, tmp_path):
        from app.indexing.vector_store import _resolve_local_sentence_transformers_path as resolve

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        assert resolve(str(model_dir), str(tmp_path)) == str(model_dir)

        repo = tmp_path / "hub" / "models--org--model" / "snapshots" / "abc123"
        repo.mkdir(parents=True)
        assert resolve("org/model", str(tmp_path)) == str(repo)

        assert resolve("org/absent", str(tmp_path)) == "org/absent"


class TestChunkerTable:
    """Stratégie TABLE de l'AdaptiveChunker (app/ingestion/chunker.py)."""

    def test_table_strategy(self):
        from pathlib import Path

        from app.ingestion.chunker import AdaptiveChunker, ChunkStrategy
        from app.ingestion.parser import ExtractionMethod, ParsedDocument, ParsedPage

        text = "| Col A | Col B |\n|---|---|\n| val1 | val2 |\n| val3 | val4 |"
        page = ParsedPage(
            content=text,
            source_file="tableau.md",
            page_number=1,
            extraction_method=ExtractionMethod.TEXT,
            file_extension=".md",
        )
        doc = ParsedDocument(source_path=Path("tableau.md"), pages=[page])

        chunks = AdaptiveChunker().chunk_document(doc)

        assert chunks
        assert any(c.strategy == ChunkStrategy.TABLE for c in chunks)
        assert all("has_header" in c.metadata for c in chunks if c.strategy == ChunkStrategy.TABLE)
