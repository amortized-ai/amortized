"""Fetch model pricing from OpenRouter's public API and output as CSV/JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_models() -> list[dict]:
    req = urllib.request.Request(MODELS_URL, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["data"]


def extract_pricing(models: list[dict]) -> list[dict]:
    rows = []
    for m in models:
        p = m.get("pricing", {})
        prompt_cost = p.get("prompt", "0")
        completion_cost = p.get("completion", "0")
        if prompt_cost == "-1" or completion_cost == "-1":
            continue

        prompt_per_m = float(prompt_cost) * 1_000_000
        completion_per_m = float(completion_cost) * 1_000_000

        rows.append(
            {
                "id": m["id"],
                "name": m.get("name", ""),
                "context_length": m.get("context_length", 0),
                "prompt_cost_per_1m": round(prompt_per_m, 4),
                "completion_cost_per_1m": round(completion_per_m, 4),
                "input_cache_read": p.get("input_cache_read"),
                "input_cache_write": p.get("input_cache_write"),
                "image_cost": p.get("image"),
                "modality": m.get("architecture", {}).get("modality", ""),
            }
        )

    rows.sort(key=lambda r: r["prompt_cost_per_1m"])
    return rows


def write_csv(rows: list[dict], out) -> None:
    if not rows:
        return
    writer = csv.DictWriter(out, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenRouter model pricing")
    parser.add_argument(
        "-f",
        "--format",
        choices=["csv", "json"],
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file (default: stdout)",
    )
    args = parser.parse_args()

    models = fetch_models()
    rows = extract_pricing(models)

    if args.output:
        dest = open(args.output, "w")
    else:
        dest = sys.stdout

    try:
        if args.format == "json":
            json.dump(rows, dest, indent=2)
            dest.write("\n")
        else:
            write_csv(rows, dest)
    finally:
        if dest is not sys.stdout:
            dest.close()

    if args.output:
        print(f"Wrote {len(rows)} models to {args.output}", file=sys.stderr)
    else:
        print(f"\n# {len(rows)} models (sorted by prompt cost)", file=sys.stderr)


if __name__ == "__main__":
    main()
