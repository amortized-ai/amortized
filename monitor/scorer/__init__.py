"""Monitor scorer package.

Offline scoring of monitor JSONL logs against a use-case checklist. Split out of
the original ``process_monitor_logs.py`` so the pieces are importable and
testable:

- ``model``    — the ``Run`` dataclass, log/checklist loading.
- ``matchers`` — the deterministic matcher registry (``auto`` rows) + ``score_run``.
- ``requires`` — row-level ``requires`` precondition registry.
- ``judge``    — the optional ``--llm-judge`` pass.
- ``report``   — the human-readable per-run + comparison output and review CSV I/O.

``monitor/scripts/process_monitor_logs.py`` is now a thin CLI over this package.
"""
