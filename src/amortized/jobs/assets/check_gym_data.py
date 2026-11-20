"""Check GRPO rows against the NeMo Gym verifier before training starts.

Runs in the training container after prepare_grpo_data.py; the gym sidecar is already up
(the pod starts the job container only once the sidecar listens). Each of the first N rows
is scored with an empty answer through gym_reward.compute_score — the exact request training
sends — so the verifier's own request schema validates the row's fields. A rejected row stops
the job here, in seconds, instead of after the model and vLLM have loaded.

Usage: python3 check_gym_data.py <grpo_data.jsonl> [rows]
"""

from __future__ import annotations

import json
import sys

from gym_reward import compute_score


def check(path: str, limit: int) -> int:
    checked = 0
    with open(path) as f:
        for index, line in enumerate(f):
            if checked >= limit:
                break
            if not line.strip():
                continue
            ground_truth = json.loads(line)["ground_truth"]
            try:
                compute_score("gym", "", ground_truth)
            except RuntimeError as exc:
                raise SystemExit(
                    f"check_gym_data: row {index} does not fit the gym verifier — fix the"
                    f" dataset's fields: {exc}"
                ) from exc
            checked += 1
    if checked == 0:
        raise SystemExit(f"check_gym_data: {path} has no rows")
    return checked


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    n = check(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    print(f"check_gym_data: the gym verifier accepted the first {n} rows")
