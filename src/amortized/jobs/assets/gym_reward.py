"""GRPO reward that scores each rollout with a NeMo Gym verifier sidecar.

The training pod runs the Gym verifier (ghcr.io/amortized-ai/gym) as a sidecar on
localhost:8000. ``thub`` resolves ``reward_fn: gym_reward.compute_score`` and training-hub
copies ONLY this function's source into verl's reward module, so the function must stay
self-contained: imports inside, no module-level helpers or constants.

``ground_truth`` is the JSON string prepare_grpo_data.py writes for each row::

    {"prompt": "<question shown to the model>", "fields": {<verifier request fields>}}

e.g. mcqa: ``"fields": {"expected_answer": "B", "options": [{"A": "..."}, {"B": "..."}]}``.
"""

from __future__ import annotations


# Builtin-only annotations: the copied source runs without `from __future__ import annotations`.
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: object = None,
    **kwargs: object,
) -> float:
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    try:
        row = json.loads(ground_truth)
        fields = dict(row["fields"])
        prompt = str(row.get("prompt") or "")
    except (TypeError, ValueError, KeyError) as exc:
        print(f"gym_reward: unreadable ground_truth ({exc!r}); scoring 0.0: {ground_truth!r:.200}")
        return 0.0

    body = {
        **fields,
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
        # Minimal OpenAI Responses object: only the output text is scored, the rest is
        # required by Gym's request schema.
        "response": {
            "id": "resp_amortized",
            "created_at": 0,
            "model": "policy",
            "object": "response",
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "output": [
                {
                    "id": "msg_amortized",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": str(solution_str), "annotations": []}
                    ],
                }
            ],
        },
    }
    url = os.environ.get("GYM_VERIFIER_URL", "http://localhost:8000/verify")
    data = json.dumps(body).encode()
    last_error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(
            url, data=data, headers={"content-type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                return float(json.load(resp)["reward"])
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                # A rejected request is a bug in this payload, not a flaky sidecar: it would
                # fail every rollout, so stop instead of training on all-zero rewards.
                detail = exc.read()[:500].decode(errors="replace")
                raise RuntimeError(
                    f"gym verifier rejected the request ({exc.code}): {detail}"
                ) from exc
            last_error = exc
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            last_error = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"gym verifier at {url} failed after 4 attempts: {last_error!r}")
