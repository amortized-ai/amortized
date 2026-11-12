"""Eval Hub execution boundary. Product APIs and approval remain in Amortized."""

SERVER_LIMIT = "This can't be fixed in the evaluation request"


def server_limit(what: str, reason: str) -> list[str]:
    """A refusal only the server's configuration can lift, worded so Morty relays it."""
    return [
        f"This server isn't set up to run {what}: {reason}. {SERVER_LIMIT}; "
        "an administrator must change the server's configuration."
    ]
