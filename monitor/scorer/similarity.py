"""Lexical text similarity for the option-redundancy matchers.

A cheap, dependency-free deterministic pre-filter (token-set Jaccard) that flags
clearly-duplicate option text / re-asked questions. The semantic/paraphrase cases
stay with the ``llm_judge`` rows (``no_duplicate_options`` / ``no_repeated_decisions``),
which are the backstop. An optional embedding path can be layered later behind a
flag; the matchers only depend on ``similarity`` + ``SIM_HIGH`` / ``SIM_GRAY``.
"""

from __future__ import annotations

import re

# >= SIM_HIGH   -> treat as the same (deterministic `wrong`).
# [SIM_GRAY, SIM_HIGH) -> uncertain; defer to the llm_judge backstop (`review`).
# < SIM_GRAY    -> distinct.
# 0.7 token-set overlap is a high bar for a lexical pre-filter: it fires only on
# "the same choice worded differently" (an extra word, a trailing clause), while
# genuinely different options (differ by a number, a model family) land in the gray
# band or below and are left to the llm_judge backstop.
SIM_HIGH = 0.7
SIM_GRAY = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")


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


def worst_pair(texts: list[str]) -> float:
    """Highest pairwise similarity among the texts (0.0 if fewer than two)."""
    hi = 0.0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            hi = max(hi, similarity(texts[i], texts[j]))
    return hi
