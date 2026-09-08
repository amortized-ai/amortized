"""Wait for an HTTP health endpoint to return 200. Exits nonzero on timeout.

Usage: wait_for_health.py <url> <timeout_seconds>
"""

import sys
import time
import urllib.request


def main() -> None:
    url, timeout_s = sys.argv[1], float(sys.argv[2])
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=2).status == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    print(f"health check timed out after {timeout_s}s: {url}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
