"""Tests for the optional semantic embedding tier (--embeddings) of the
duplicate/repeated-option checks. Uses a fake embedder — no model download."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from monitor.scorer import similarity
from monitor.scorer.matchers import score_auto
from monitor.scorer.model import Run

# concept vectors: paraphrases map to the same vector (cosine 1.0), different
# concepts are orthogonal.
_VECS: dict[str, list[float]] = {
    "Use the whole dataset": [1.0, 0.0, 0.0],
    "Use the entire corpus": [1.0, 0.0, 0.0],  # paraphrase, lexically distinct
    "Train a Qwen model": [0.0, 1.0, 0.0],
    "Generate 1000 examples": [0.0, 0.0, 1.0],
    "Generate 2000 examples": [0.0, 0.0, 1.0],  # same concept, numeric-only diff
}


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [_VECS.get(t, [0.0, 0.0, 0.0]) for t in texts]


@pytest.fixture()
def embeddings_on() -> Iterator[None]:
    similarity.set_embedder(_fake_embed)
    try:
        yield
    finally:
        similarity.set_embedder(None)  # never leak global state into other tests


def _po(question: str, options: list[str]) -> dict[str, Any]:
    return {"tool": "present_options", "role": "sdg", "question": question, "options": options}


def _run(*tool_calls: dict[str, Any]) -> Run:
    return Run(
        session_id="t",
        turns=[{"ts": "2026-01-01T00:00:00", "role": "sdg", "tool_calls": list(tool_calls)}],
    )


def test_embeddings_disabled_by_default() -> None:
    assert similarity.embeddings_enabled() is False
    assert similarity.embedding_similarity("a", "b") is None


def test_paraphrase_is_review_lexically_but_wrong_with_embeddings(
    embeddings_on: None,
) -> None:
    # lexically these share only "use the" -> gray (review) without embeddings;
    # the fake embedder maps them to the same vector -> wrong with embeddings.
    run = _run(_po("Pick one", ["Use the whole dataset", "Use the entire corpus"]))
    assert score_auto(run, {"type": "duplicate_options"}) == "wrong"


def test_paraphrase_missed_by_lexical_without_embeddings() -> None:
    run = _run(_po("Pick one", ["Use the whole dataset", "Use the entire corpus"]))
    # lexically these overlap too little (0.33 < gray) -> lexical calls them distinct
    # (met). This is exactly the paraphrase the embedding tier is meant to catch.
    assert score_auto(run, {"type": "duplicate_options"}) == "met"


def test_numeric_only_diff_stays_distinct_even_with_embeddings(embeddings_on: None) -> None:
    # same concept vector, but a numeric-only difference is a distinct CHOICE
    run = _run(_po("How many?", ["Generate 1000 examples", "Generate 2000 examples"]))
    assert score_auto(run, {"type": "duplicate_options"}) == "met"


def test_distinct_concepts_met_with_embeddings(embeddings_on: None) -> None:
    run = _run(_po("Pick", ["Use the whole dataset", "Train a Qwen model"]))
    assert score_auto(run, {"type": "duplicate_options"}) == "met"


def test_repeated_decision_uses_embeddings(embeddings_on: None) -> None:
    # two prompts in the SAME segment asking the paraphrased same thing -> wrong
    run = _run(
        _po("Use the whole dataset", ["a", "b"]),
        _po("Use the entire corpus", ["c", "d"]),
    )
    assert score_auto(run, {"type": "repeated_decision"}) == "wrong"


def test_embedding_similarity_cosine(embeddings_on: None) -> None:
    assert similarity.embedding_similarity("Use the whole dataset", "Use the entire corpus") == 1.0
    assert similarity.embedding_similarity("Use the whole dataset", "Train a Qwen model") == 0.0
