"""Smoke-test every verifier in the image the way a training pod runs it.

For each name: start run_verifier.sh (GYM_SERVER=<name>), wait for :8000, POST the first row of
the verifier's own example data with its recorded answer (example_rollouts.jsonl, else a stub
text answer), and require HTTP 200 with a numeric reward. Prints one JSON result per verifier
and a summary; exits non-zero if any failed.

Usage: python test_verifiers.py [name ...]   (default: every name in verifiers.txt)
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

GYM = Path("/opt/gym/resources_servers")
LIST = Path("/opt/amortized-gym/verifiers.txt")


def _drop_nulls(value):
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def _text_response(text: str) -> dict:
    return {
        "id": "resp_test", "created_at": 0, "model": "policy", "object": "response",
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "output": [{
            "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
    }


def _request(name: str) -> dict:
    data = GYM / name / "data"
    row = json.loads((data / "example.jsonl").read_text().splitlines()[0])
    body = _drop_nulls(row)
    response = None
    rollouts = data / "example_rollouts.jsonl"
    if rollouts.exists():
        recorded = json.loads(rollouts.read_text().splitlines()[0]).get("response")
        texts = [
            c.get("text", "")
            for item in (recorded or {}).get("output", [])
            if item.get("type") == "message"
            for c in item.get("content") or []
            if c.get("type") == "output_text"
        ]
        if texts:
            response = _text_response("".join(texts))
    body["response"] = response or _text_response("test answer")
    body.setdefault("responses_create_params", {"input": [{"role": "user", "content": "test"}]})
    return body


def _wait_port(proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", 8000)) == 0:
                return True
        time.sleep(0.5)
    return False


def test(name: str) -> dict:
    log = Path(f"/tmp/verifier-{name}.log")
    env = {**os.environ, "GYM_SERVER": name}
    with log.open("w") as out:
        proc = subprocess.Popen(
            ["run_verifier.sh"], env=env, stdout=out, stderr=subprocess.STDOUT
        )
    result = {"name": name, "ok": False}
    try:
        started = time.time()
        if not _wait_port(proc, 90):
            result["error"] = "did not start: " + log.read_text()[-400:].replace("\n", " | ")
            return result
        result["startup_s"] = round(time.time() - started, 1)
        request = urllib.request.Request(
            "http://127.0.0.1:8000/verify",
            data=json.dumps(_request(name)).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as resp:
                reward = json.load(resp).get("reward")
        except urllib.error.HTTPError as exc:
            result["error"] = f"HTTP {exc.code}: {exc.read()[:300].decode(errors='replace')}"
            return result
        result["reward"] = reward
        result["ok"] = isinstance(reward, (int, float))
        if not result["ok"]:
            result["error"] = f"non-numeric reward {reward!r}"
        return result
    except Exception as exc:  # report, keep testing the rest
        result["error"] = repr(exc)[:300]
        return result
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    names = sys.argv[1:] or [
        line.strip() for line in LIST.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    results = [test(n) for n in names]
    for r in results:
        print(json.dumps(r))
    passed = [r["name"] for r in results if r["ok"]]
    print(f"SUMMARY {len(passed)}/{len(results)} passed")
    print("PASSED " + " ".join(passed))
    sys.exit(0 if len(passed) == len(results) else 1)
