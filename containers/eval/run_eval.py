#!/usr/bin/env python3
"""Eval runner — score ONE model endpoint on an eval dataset (absolute).

Reads a config.json (delivered to /amortized/config.json by the control plane):

    {
      "eval_data_path": "/amortized/work/eval_data/generated_data",
      "endpoints": {
        "model": {"base_url": "http://host:8000/v1", "model": "m",
                  "api_key_env": "EVAL_MODEL_API_KEY"},
        "judge": {"base_url": "...", "model": "...", "api_key_env": "EVAL_JUDGE_API_KEY"}
      },
      "metrics": ["exact_match", "format_validity"],
      "rubric": [{"name": "accuracy", "description": "Facts match the reference"}],
      "max_samples": 200,
      "judge_max_samples": 0,  # 0 = judge all samples
      "temperature": 0.0,
      "output_dir": "/amortized/work/results"
    }

One model per eval job. Structural metrics (exact_match, format_validity, ...)
are computed against the held-out reference answer. When a rubric + judge are
provided, the judge scores the model's response against the reference on each
criterion on an absolute 0-10 scale (reported normalized to 0-1), averaged
over the dataset — NO pairwise win-rate comparison.

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
import re
from pathlib import Path
from typing import Any

import httpx

MAX_PARALLEL = 16
REQUEST_TIMEOUT = 300.0
MAX_RETRIES = 3
SCORE_SCALE = 10  # judge scores each criterion 0..SCORE_SCALE; reported /SCALE


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


SCORE_SYSTEM_PROMPT = f"""\
You are an impartial judge scoring a single candidate response against a \
reference (gold) answer.
Given the task prompt, the reference answer, the candidate response, and a \
list of evaluation criteria, score the candidate on EACH criterion from 0 to \
{SCORE_SCALE}, where 0 means the criterion is not met at all (wrong, missing, \
or contradicts the reference) and {SCORE_SCALE} means it is fully met (matches \
the reference in correctness and completeness).
Judge only against the reference; do not reward style beyond what the \
criterion asks. Respond with ONLY a JSON object mapping every criterion name \
to an integer 0-{SCORE_SCALE}, e.g. {{"accuracy": 8, "reasoning_quality": 5}}.
"""


def parse_scores(raw: str, criteria: list[str]) -> dict[str, float | None]:
    """Extract per-criterion 0..SCORE_SCALE scores from a judge response.

    Missing/unparseable criteria default to None so a partial judge response
    never crashes aggregation.
    """
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(match.group(0)) if match else {}
    except ValueError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    scores: dict[str, float | None] = {}
    for criterion in criteria:
        value = data.get(criterion)
        try:
            num = float(value)
        except (TypeError, ValueError):
            scores[criterion] = None
            continue
        num = max(0.0, min(float(SCORE_SCALE), num))
        scores[criterion] = num
    return scores


async def score_one(
    client: httpx.AsyncClient,
    judge: dict[str, Any],
    api_key: str,
    prompt: list[dict[str, str]],
    reference: str,
    output: str,
    rubric: list[dict[str, str]],
) -> dict[str, float | None]:
    """Score one response against the reference on each rubric criterion."""
    lines = "\n".join(f"- {c['name']}: {c.get('description', '')}" for c in rubric)
    user_content = (
        f"## Task prompt\n{json.dumps(prompt)}\n\n"
        f"## Reference answer\n{reference}\n\n"
        f"## Evaluation criteria\n{lines}\n\n"
        f"## Candidate response\n{output or '(empty)'}\n\n"
        f"Score the candidate on each criterion from 0 to {SCORE_SCALE}."
    )
    names = [c["name"] for c in rubric]
    try:
        raw = await chat_completion(
            client,
            judge,
            [
                {"role": "system", "content": SCORE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=1024,
            api_key=api_key,
        )
        return parse_scores(raw, names)
    except Exception:
        return {name: None for name in names}


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
    max_samples = int(config.get("max_samples", 200))
    # 0 = judge every sample (the default); a positive value caps judge
    # cost/latency for large evals.
    judge_max_samples = int(config.get("judge_max_samples", 0))
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
        raw = await eval_endpoint(
            client, endpoints["model"], samples, api_keys.get("model", ""), temperature
        )
        outputs = [r["output"] for r in raw]
        results["model"] = structural_metrics(
            outputs,
            [s["reference"] for s in samples],
            [r["error"] for r in raw],
        )

        # Absolute per-criterion scoring against the reference (no pairing).
        rubric = [c for c in (config.get("rubric") or []) if isinstance(c, dict) and c.get("name")]
        sample_scores: list[dict[str, float | None]] = [{} for _ in samples]
        if "judge" in endpoints and rubric:
            semaphore = asyncio.Semaphore(MAX_PARALLEL)

            async def score_idx(idx: int) -> tuple[int, dict[str, float | None]]:
                async with semaphore:
                    s = await score_one(
                        client,
                        endpoints["judge"],
                        api_keys.get("judge", ""),
                        samples[idx]["prompt"],
                        samples[idx]["reference"],
                        outputs[idx],
                        rubric,
                    )
                    return idx, s

            limit = len(samples) if judge_max_samples <= 0 else min(judge_max_samples, len(samples))
            scorable = [
                i for i in range(limit) if samples[i]["reference"].strip() and not raw[i]["error"]
            ]
            scored = await asyncio.gather(*(score_idx(i) for i in scorable))
            for idx, s in scored:
                sample_scores[idx] = s

            scores_out: dict[str, float | None] = {}
            scores_n: dict[str, int] = {}
            for c in rubric:
                name = c["name"]
                vals = [
                    sample_scores[i][name]
                    for i in scorable
                    if sample_scores[i].get(name) is not None
                ]
                scores_n[name] = len(vals)
                scores_out[name] = (
                    round(sum(vals) / len(vals) / SCORE_SCALE, 4) if vals else None
                )
            results["scores"] = scores_out
            results["scores_n"] = scores_n
            results["num_scored"] = len(scorable)

    rows = []
    for i, sample in enumerate(samples):
        rows.append(
            {
                "sample_index": i,
                "prompt": sample["prompt"],
                "reference": sample["reference"],
                "output": outputs[i],
                "scores": sample_scores[i],
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

    # Hard-fail if the model mostly errored — metrics would be meaningless
    if summary["results"]["model"]["error_rate"] > 0.5:
        raise SystemExit(
            f"model errored on {summary['results']['model']['error_rate']:.0%} of samples"
        )


if __name__ == "__main__":
    main()
