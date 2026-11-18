"""Eval Hub execution boundary. Product APIs and approval remain in Amortized."""

from typing import Any

import amortized.config as config_mod

SERVER_LIMIT = "This can't be fixed in the evaluation request"


def server_limit(what: str, reason: str) -> list[str]:
    """A refusal only the server's configuration can lift, worded so Morty relays it."""
    return [
        f"This server isn't set up to run {what}: {reason}. {SERVER_LIMIT}; "
        "an administrator must change the server's configuration."
    ]


def accepts_new_jobs() -> bool:
    """Hub may take new evaluations. Jobs already on Hub are followed whenever its URL is set."""
    settings = config_mod.settings
    return bool(settings.evalhub_url) and settings.evalhub_accept_new_jobs


def runs_on_hub(config: dict[str, Any]) -> bool:
    """Where a new evaluation runs: on Hub, or natively.

    A classifier head trained with use_peft (eval_on_gpu, stamped by the API) needs a GPU,
    and Hub's task adapter runs on CPU only, so it evaluates natively.
    """
    return accepts_new_jobs() and not config.get("eval_on_gpu")
