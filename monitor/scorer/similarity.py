"""Text similarity for the option-redundancy matchers — lexical + optional semantic.

Two tiers:
- **Lexical** (always on, dependency-free): token-set Jaccard with `SIM_HIGH`/`SIM_GRAY`
  bands — flags clearly-duplicate option text / re-asked questions.
- **Semantic embeddings** (optional, `--embeddings`): when an embedder is configured,
  cosine over embeddings resolves pairs the lexical tier is unsure about and catches
  paraphrase duplicates (same meaning, different words) deterministically.

The `llm_judge` rows (`no_duplicate_options` / `no_repeated_decisions`) remain the
final backstop regardless.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable

# >= SIM_HIGH   -> treat as the same (deterministic `wrong`).
# [SIM_GRAY, SIM_HIGH) -> uncertain; defer to the llm_judge backstop (`review`).
# < SIM_GRAY    -> distinct.
# 0.7 token-set overlap is a high bar for a lexical pre-filter: it fires only on
# "the same choice worded differently" (an extra word, a trailing clause), while
# genuinely different options (differ by a number, a model family) land in the gray
# band or below and are left to the llm_judge backstop.
SIM_HIGH = 0.7
SIM_GRAY = 0.5
# Cosine threshold for the embedding tier: >= means "same meaning".
SIM_EMBED_HIGH = 0.85

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")

# Pluggable embedder: list[str] -> list[vector]. None = embeddings disabled.
Embedder = Callable[[list[str]], list[list[float]]]
_EMBEDDER: Embedder | None = None
_EMBED_CACHE: dict[str, list[float]] = {}


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Token-set Jaccard of two strings (0.0 to 1.0). Two blank strings -> 0."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def differs_only_numerically(a: str, b: str) -> bool:
    """True when two strings differ ONLY by numeric tokens (e.g. "Generate 1000"
    vs "Generate 2000"). Such options are distinct *choices*, not duplicates, so
    the redundancy matchers must not flag or defer them."""
    diff = _tokens(a) ^ _tokens(b)
    return bool(diff) and all(_NUM_RE.match(t) for t in diff)


# --------------------------------------------------------------------------- #
# Embedding tier (optional)
# --------------------------------------------------------------------------- #


def set_embedder(fn: Embedder | None) -> None:
    """Enable (or clear) the semantic tier. Called by the CLI for `--embeddings`."""
    global _EMBEDDER
    _EMBEDDER = fn
    _EMBED_CACHE.clear()


def embeddings_enabled() -> bool:
    return _EMBEDDER is not None


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(x * y for x, y in zip(u, v, strict=False))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    return dot / (nu * nv) if nu and nv else 0.0


def _embed(text: str) -> list[float] | None:
    if _EMBEDDER is None:
        return None
    if text not in _EMBED_CACHE:
        try:
            _EMBED_CACHE[text] = list(_EMBEDDER([text])[0])
        except Exception:
            return None
    return _EMBED_CACHE[text]


def embedding_similarity(a: str, b: str) -> float | None:
    """Cosine similarity of two strings' embeddings, or None when embeddings are
    disabled or embedding failed."""
    va, vb = _embed(a), _embed(b)
    if va is None or vb is None:
        return None
    return _cosine(va, vb)


def default_embedder(model: str = "all-MiniLM-L6-v2") -> Embedder:
    """A local sentence-transformers embedder (needs the `monitor` extra). Lazily
    loads the model once; each call encodes a batch of texts to vectors."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "--embeddings needs sentence-transformers: uv pip install 'amortized[monitor]'"
        ) from exc
    st_model = SentenceTransformer(model)

    def embed(texts: list[str]) -> list[list[float]]:
        return [list(v) for v in st_model.encode(texts, normalize_embeddings=False)]

    return embed
