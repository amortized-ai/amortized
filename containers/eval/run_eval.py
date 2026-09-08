#!/usr/bin/env python3
"""Eval runner — compare base and tuned model endpoints on an eval dataset.

Reads a config.json (delivered to /amortized/config.json by the control plane):

    {
      "eval_data_path": "/amortized/work/eval_data/generated_data",
      "endpoints": {
        "base":  {"base_url": "http://host:8000/v1", "model": "m",
                  "api_key_env": "EVAL_BASE_API_KEY"},
        "tuned": {"base_url": "http://host:8001/v1", "model": "m",
                  "api_key_env": "EVAL_TUNED_API_KEY"},
        "judge": {"base_url": "...", "model": "...", "api_key_env": "EVAL_JUDGE_API_KEY"}
      },
      "metrics": ["exact_match", "format_validity", "judge_win_rate"],
      "rubric": [{"name": "accuracy", "description": "Facts match the reference"}],
      "max_samples": 200,
      "judge_max_samples": 100,
      "temperature": 0.0,
      "output_dir": "/amortized/work/results"
    }

Dataset format: jsonl or parquet with a `messages` column (list of
{role, content} dicts). The trailing assistant message is held out as the
reference answer; the rest is the prompt.

Outputs metrics.json (aggregates) and results.jsonl (per-sample) to output_dir.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import httpx

MAX_PARALLEL = 16
REQUEST_TIMEOUT = 300.0
MAX_RETRIES = 3


def load_records(eval_data_path: str) -> list[dict[str, Any]]:
    files: list[str] = []
    for pattern in ("**/*.jsonl", "**/*.parquet"):
        files.extend(glob.glob(os.path.join(eval_data_path, pattern), recursive=True))
    files.sort()

    records: list[dict[str, Any]] = []
    for path in files:
        if path.endswith(".jsonl"):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        elif path.endswith(".parquet"):
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            records.extend(table.to_pylist())

    if not records:
        raise SystemExit(f"No .jsonl or .parquet records found under {eval_data_path}")
    return records


def split_prompt_reference(record: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
    messages = record.get("messages") or []
    messages = [m for m in messages if isinstance(m, dict) and m.get("content")]
    if not messages:
        raise ValueError("record has no non-empty messages")

    reference = ""
    prompt = messages
    if messages and messages[-1].get("role") == "assistant":
        reference = str(messages[-1].get("content", ""))
        prompt = messages[:-1]
    if not prompt:
        raise ValueError("record has no prompt messages after removing reference")
    return prompt, reference


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_json(text: str) -> Any:
    return json.loads(text)


def _try_json(text: str) -> bool:
    try:
        _extract_json(text)
        return True
    except (ValueError, TypeError):
        return False


async def chat_completion(
    client: httpx.AsyncClient,
    endpoint: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    api_key: str,
) -> str:
    url = endpoint["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {
        "model": endpoint["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return str(data["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"endpoint {endpoint['model']}: {last_error}")


async def eval_endpoint(
    client: httpx.AsyncClient,
    endpoint: dict[str, Any],
    samples: list[dict[str, Any]],
    api_key: str,
    temperature: float,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(MAX_PARALLEL)

    async def run_one(sample: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            try:
                output = await chat_completion(
                    client,
                    endpoint,
                    sample["prompt"],
                    temperature=temperature,
                    max_tokens=4096,
                    api_key=api_key,
                )
                return {"output": output, "error": ""}
            except Exception as exc:
                return {"output": "", "error": str(exc)}

    return list(await asyncio.gather(*(run_one(s) for s in samples)))


JUDGE_SYSTEM_PROMPT = """\
You are an impartial judge comparing two candidate responses to a task.
Given the task prompt, the reference (gold) answer, and two candidates \
labeled Candidate A and Candidate B, decide which candidate is closer to \
the reference in correctness and completeness.
Respond with ONLY a JSON object: {"winner": "A"}, {"winner": "B"}, \
or {"winner": "tie"}.
"""

RUBRIC_SYSTEM_PROMPT = """\
You are an impartial judge comparing two candidate responses to a task.
Given the task prompt, the reference (gold) answer, two candidates \
labeled Candidate A and Candidate B, and a list of evaluation criteria, \
decide which candidate is better for EACH criterion.
Respond with ONLY a JSON object mapping every criterion name to "A", \
"B", or "tie", e.g. {"accuracy": "A", "tone": "tie"}.
"""


def parse_rubric_verdicts(raw: str, criteria: list[str]) -> dict[str, str]:
    """Extract per-criterion A/B/tie verdicts from a judge response.

    Unknown or missing criteria default to "tie" so a partial judge
    response never crashes aggregation.
    """
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    verdicts: dict[str, str] = {}
    for criterion in criteria:
        value = data.get(criterion, "tie")
        verdicts[criterion] = value if value in ("A", "B", "tie") else "tie"
    return verdicts


async def judge_one(
    client: httpx.AsyncClient,
    judge: dict[str, Any],
    api_key: str,
    prompt: list[dict[str, str]],
    reference: str,
    output_a: str,
    output_b: str,
    idx: int,
    rubric: list[dict[str, str]] | None = None,
) -> str | dict[str, str]:
    """Judge one sample. Returns "A"/"B"/"tie", or with a rubric a
    {criterion_name: "A"/"B"/"tie"} mapping."""
    criteria_block = ""
    if rubric:
        lines = "\n".join(
            f"- {c['name']}: {c.get('description', '')}" for c in rubric
        )
        criteria_block = f"## Evaluation criteria\n{lines}\n\n"
        question = "Which candidate is better for each criterion?"
        system = RUBRIC_SYSTEM_PROMPT
    else:
        question = "Which candidate is closer to the reference?"
        system = JUDGE_SYSTEM_PROMPT

    user_content = (
        f"## Task prompt\n"
        f"{json.dumps(prompt)}\n\n"
        f"## Reference answer\n{reference}\n\n"
        f"{criteria_block}"
        f"## Candidate A\n{output_a or '(empty)'}\n\n"
        f"## Candidate B\n{output_b or '(empty)'}\n\n"
        f"{question}"
    )
    try:
        raw = await chat_completion(
            client,
            judge,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
            api_key=api_key,
        )
        if rubric:
            return parse_rubric_verdicts(raw, [c["name"] for c in rubric])
        match = re.search(r'"winner"\s*:\s*"(A|B|tie)"', raw)
        return match.group(1) if match else "tie"
    except Exception:
        if rubric:
            return {c["name"]: "tie" for c in rubric}
        return "tie"


def structural_metrics(
    outputs: list[str],
    references: list[str],
    errors: list[str],
) -> dict[str, Any]:
    n = len(outputs)
    succeeded = [i for i in range(n) if not errors[i]]
    empty = sum(1 for i in succeeded if not outputs[i].strip())

    ref_present = [i for i in succeeded if references[i].strip()]
    exact = sum(1 for i in ref_present if _normalize(outputs[i]) == _normalize(references[i]))

    json_refs = [i for i in ref_present if _try_json(references[i])]
    json_valid = sum(1 for i in json_refs if _try_json(outputs[i]))

    return {
        "num_samples": n,
        "num_succeeded": len(succeeded),
        "error_rate": round(1 - len(succeeded) / n, 4) if n else 1.0,
        "empty_rate": round(empty / len(succeeded), 4) if succeeded else 1.0,
        "exact_match": round(exact / len(ref_present), 4) if ref_present else None,
        "exact_match_n": len(ref_present),
        "format_validity": round(json_valid / len(json_refs), 4) if json_refs else None,
        "format_validity_n": len(json_refs),
    }


async def run(config: dict[str, Any]) -> dict[str, Any]:
    endpoints = config["endpoints"]
    metrics_requested = set(config.get("metrics", []))
    max_samples = int(config.get("max_samples", 200))
    judge_max_samples = int(config.get("judge_max_samples", 100))
    temperature = float(config.get("temperature", 0.0))
    output_dir = Path(config.get("output_dir", "/amortized/work/results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(config["eval_data_path"])
    samples: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        if len(samples) >= max_samples:
            break
        try:
            prompt, reference = split_prompt_reference(record)
        except ValueError:
            skipped += 1
            continue
        samples.append({"prompt": prompt, "reference": reference})
    if not samples:
        raise SystemExit("No usable samples (records missing 'messages' column?)")

    api_keys = {
        name: os.environ.get(ep.get("api_key_env", ""), "") for name, ep in endpoints.items()
    }

    results: dict[str, Any] = {"num_records": len(records), "num_skipped": skipped}

    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT)) as client:
        per_model: dict[str, dict[str, Any]] = {}
        outputs_by_model: dict[str, list[str]] = {}
        for name in ("base", "tuned"):
            raw = await eval_endpoint(
                client, endpoints[name], samples, api_keys.get(name, ""), temperature
            )
            outputs_by_model[name] = [r["output"] for r in raw]
            per_model[name] = structural_metrics(
                [r["output"] for r in raw],
                [s["reference"] for s in samples],
                [r["error"] for r in raw],
            )
            results[name] = per_model[name]

        judged = 0
        rubric = [c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")]
        if "judge" in endpoints and "judge_win_rate" in metrics_requested:
            semaphore = asyncio.Semaphore(MAX_PARALLEL)

            async def judge_pair(idx: int) -> str | dict[str, str]:
                async with semaphore:
                    # Deterministic side randomization to cancel position bias
                    a_first = random.Random(idx).random() < 0.5
                    first, second = ("base", "tuned") if a_first else ("tuned", "base")
                    verdict = await judge_one(
                        client,
                        endpoints["judge"],
                        api_keys.get("judge", ""),
                        samples[idx]["prompt"],
                        samples[idx]["reference"],
                        outputs_by_model[first][idx],
                        outputs_by_model[second][idx],
                        idx,
                        rubric=rubric or None,
                    )
                    if isinstance(verdict, dict):
                        return {
                            criterion: _map_label(label, first, second)
                            for criterion, label in verdict.items()
                        }
                    return _map_label(verdict, first, second)

            def _map_label(label: str, first: str, second: str) -> str:
                if label == "tie":
                    return "tie"
                return first if label == "A" else second

            limit = min(judge_max_samples, len(samples))
            judgable = [i for i in range(limit) if samples[i]["reference"].strip()]
            verdicts = list(await asyncio.gather(*(judge_pair(i) for i in judgable)))
            judged = len(verdicts)

            if rubric:
                criteria_stats: dict[str, dict[str, int]] = {
                    c["name"]: {"tuned": 0, "base": 0, "tie": 0} for c in rubric
                }
                for v in verdicts:
                    for criterion, winner in v.items():
                        criteria_stats[criterion][winner] += 1
                criteria_out: dict[str, Any] = {}
                for name, stats in criteria_stats.items():
                    n = sum(stats.values())
                    criteria_out[name] = {
                        "tuned_wins": stats["tuned"],
                        "base_wins": stats["base"],
                        "ties": stats["tie"],
                        "win_rate": (
                            round((stats["tuned"] + 0.5 * stats["tie"]) / n, 4)
                            if n
                            else None
                        ),
                    }
                win_rates = [
                    c["win_rate"] for c in criteria_out.values() if c["win_rate"] is not None
                ]
                results["judge"] = {
                    "num_judged": judged,
                    "criteria": criteria_out,
                    "win_rate": round(sum(win_rates) / len(win_rates), 4) if win_rates else None,
                }
            else:
                tuned_wins = sum(1 for v in verdicts if v == "tuned")
                ties = sum(1 for v in verdicts if v == "tie")
                results["judge"] = {
                    "num_judged": judged,
                    "tuned_wins": tuned_wins,
                    "base_wins": judged - tuned_wins - ties,
                    "ties": ties,
                    "win_rate": (round((tuned_wins + 0.5 * ties) / judged, 4) if judged else None),
                }

    rows = []
    for i, sample in enumerate(samples):
        rows.append(
            {
                "sample_index": i,
                "prompt": sample["prompt"],
                "reference": sample["reference"],
                "output_base": outputs_by_model["base"][i],
                "output_tuned": outputs_by_model["tuned"][i],
            }
        )
    with open(output_dir / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {"results": results}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)

    summary = asyncio.run(run(config))

    # Hard-fail if either endpoint mostly errored — metrics would be meaningless
    for name in ("base", "tuned"):
        if summary["results"][name]["error_rate"] > 0.5:
            raise SystemExit(
                f"endpoint '{name}' errored on "
                f"{summary['results'][name]['error_rate']:.0%} of samples"
            )


if __name__ == "__main__":
    main()
