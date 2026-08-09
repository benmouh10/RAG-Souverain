from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app_ui import (
    build_chat_message,
    build_source_payloads,
    dedupe_sources,
    get_upload_widget_key,
    is_greeting,
    reset_upload_widget_state,
    truncate_snippet,
)


def test_chat_message_keeps_sources_for_assistant_messages():
    payload = build_chat_message(
        "assistant",
        "Réponse générée",
        sources=[{"source_file": "doc.pdf", "score": 0.91, "page_number": 3}],
    )

    assert payload["role"] == "assistant"
    assert payload["content"] == "Réponse générée"
    assert payload["sources"] == [{"source_file": "doc.pdf", "score": 0.91, "page_number": 3}]


def test_build_source_payloads_uses_result_metadata():
    class FakeResult:
        def __init__(self, content, score, metadata):
            self.content = content
            self.score = score
            self.metadata = metadata

    results = [
        FakeResult("chunk one", 0.87, {"source_file": "alpha.txt", "page_number": 2}),
        FakeResult("chunk two", 0.74, {"source_file": "beta.txt", "page_number": 5}),
    ]

    payloads = build_source_payloads(results)

    assert payloads[0]["source_file"] == "alpha.txt"
    assert payloads[0]["page_number"] == 2
    assert payloads[0]["score"] == 0.87
    assert payloads[1]["source_file"] == "beta.txt"


def test_upload_widget_state_can_be_reset():
    st.session_state.clear()

    first_key = get_upload_widget_key()
    second_key = reset_upload_widget_state()

    assert first_key != second_key
    assert isinstance(second_key, int)
    assert second_key >= 1


def test_greeting_detection_skips_short_hellos():
    assert is_greeting("hey")
    assert is_greeting("Hello")
    assert is_greeting("bonjour")
    assert is_greeting("salut")
    assert is_greeting("hi")


def test_greeting_detection_rejects_real_questions():
    assert not is_greeting("quel est le prix du clavier mecanique ?")
    assert not is_greeting("hello how does the evaluation pipeline work")
    assert not is_greeting("")


def test_dedupe_sources_removes_identical_contents():
    sources = [
        {"content": "abc", "source_file": "a.pdf", "page_number": 1},
        {"content": "abc", "source_file": "a.pdf", "page_number": 1},
        {"content": "def", "source_file": "b.pdf", "page_number": 2},
    ]
    unique = dedupe_sources(sources)
    assert len(unique) == 2
    assert unique[0]["content"] == "abc"
    assert unique[1]["content"] == "def"


def test_truncate_snippet_keeps_previews_short():
    short = truncate_snippet("petit extrait")
    assert short == "petit extrait"

    long_text = "x" * 500
    truncated = truncate_snippet(long_text, 250)
    assert len(truncated) <= 250
    assert truncated.endswith("…")

    spaced = truncate_snippet(" ".join(["mot"] * 80), 50)
    assert len(spaced) <= 50
