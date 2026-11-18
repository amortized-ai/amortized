"""Shared by the release helpers: the Hugging Face dataset a benchmark scores on.

Frameworks spell the same dataset differently: legacy names ("gsm8k") redirect to
an owner/name repo ("openai/gsm8k"), and an unnamed config is HF's "default".
Recording the canonical spelling lets Amortized see two catalog entries that score
the same questions. Needs network; a repo the Hub cannot resolve is kept as written.
"""

from typing import Any

_CANONICAL: dict[str, str] = {}


def canonical_repo(repo: str) -> str:
    if not repo or repo in _CANONICAL:
        return _CANONICAL.get(repo, repo)
    try:
        from huggingface_hub import HfApi

        _CANONICAL[repo] = str(HfApi().dataset_info(repo).id)
    except Exception:
        _CANONICAL[repo] = repo
    return _CANONICAL[repo]


def dataset(task: str, repo: Any, subset: Any, splits: list[str], revision: Any) -> dict[str, Any]:
    written = str(repo or "")
    record = {
        "task": task,
        "hf_repo": canonical_repo(written) or None,
        "hf_subset": str(subset) if subset else "default",
        "splits": sorted(splits),
        "hf_revision": revision or None,
    }
    if record["hf_repo"] != (written or None):
        record["hf_repo_as_written"] = written
    return record
