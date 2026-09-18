"""Wait for the in-pod model server to answer /health.

Fails fast when the server process dies (e.g. CUDA OOM during engine
init) instead of polling until the timeout, so the eval job fails in
seconds rather than after a 15-minute wait.

Usage: python3 wait_for_endpoint.py <port> <serve_pid> [timeout_s]
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request


def _serve_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # os.kill(pid, 0) also succeeds for a zombie — the process died but
    # its parent hasn't reaped it yet (the shell that started serve and
    # is busy running this script). A zombie server is a dead server:
    # without this check, a serve OOM polls the full timeout.
    try:
        with open(f"/proc/{pid}/stat") as f:
            # Field 3 is the state; everything after the comm in parens
            # is space-separated.
            rest = f.read().rsplit(")", 1)[-1].split()
            if len(rest) >= 1 and rest[0] == "Z":
                return False
    except OSError:
        pass
    return True


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    serve_pid = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 900

    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _serve_alive(serve_pid):
            print(
                f"wait_for_endpoint: model server (pid {serve_pid}) exited"
                " before becoming healthy — see its logs above",
                flush=True,
            )
            return 2
        try:
            urllib.request.urlopen(url, timeout=5)
            return 0
        except Exception:
            time.sleep(5)
    print(
        f"wait_for_endpoint: model server did not become healthy within"
        f" {timeout}s",
        flush=True,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
