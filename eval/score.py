"""Factory eval harness for Amortized."""

import json
import re
import subprocess
import sys
from pathlib import Path

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "server"


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=300)


def check_tests() -> dict:
    """Run pytest in runtime/ and check exit code."""
    result = run_cmd([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=RUNTIME_DIR)
    passed = result.returncode == 0
    return {
        "name": "tests",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": result.stdout[-2000:] if result.stdout else result.stderr[-2000:],
    }


def check_lint() -> dict:
    """Run ruff check in runtime/."""
    result = run_cmd(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/"], cwd=RUNTIME_DIR
    )
    passed = result.returncode == 0
    return {
        "name": "lint",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": result.stdout[-2000:] if result.stdout else result.stderr[-2000:],
    }


def check_type_check() -> dict:
    """Run mypy in runtime/."""
    result = run_cmd([sys.executable, "-m", "mypy", "src/"], cwd=RUNTIME_DIR)
    passed = result.returncode == 0
    return {
        "name": "type_check",
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "details": result.stdout[-2000:] if result.stdout else result.stderr[-2000:],
    }


def check_capability_surface() -> dict:
    """Count API endpoints defined in the runtime."""
    main_py = RUNTIME_DIR / "src" / "amortized" / "main.py"
    if not main_py.exists():
        return {"name": "capability_surface", "passed": False, "score": 0.0, "details": "main.py not found"}

    content = main_py.read_text()
    endpoint_pattern = r"@app\.(get|post|put|patch|delete|websocket)\("
    endpoints = re.findall(endpoint_pattern, content)
    count = len(endpoints)

    return {
        "name": "capability_surface",
        "passed": count > 0,
        "score": min(count / 10.0, 1.0),
        "details": f"{count} endpoint(s) defined",
    }


def check_observability() -> dict:
    """Check for logging setup in the runtime."""
    main_py = RUNTIME_DIR / "src" / "amortized" / "main.py"
    if not main_py.exists():
        return {"name": "observability", "passed": False, "score": 0.0, "details": "main.py not found"}

    content = main_py.read_text()
    has_logging_import = "import logging" in content or "from logging" in content
    has_logger = "getLogger" in content or "logger" in content
    has_basic_config = "basicConfig" in content

    checks_passed = sum([has_logging_import, has_logger, has_basic_config])
    score = checks_passed / 3.0

    return {
        "name": "observability",
        "passed": checks_passed >= 2,
        "score": score,
        "details": f"logging_import={has_logging_import}, logger={has_logger}, basic_config={has_basic_config}",
    }


def main() -> None:
    checks = [
        check_tests,
        check_lint,
        check_type_check,
        check_capability_surface,
        check_observability,
    ]

    results = []
    for check in checks:
        result = check()
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['name']}: {result['score']:.2f} — {result['details'][:100]}")

    composite = sum(r["score"] for r in results) / len(results)
    print(f"\nComposite score: {composite:.2f}")

    output = {"dimensions": results, "composite_score": composite}
    print(json.dumps(output))


if __name__ == "__main__":
    main()
