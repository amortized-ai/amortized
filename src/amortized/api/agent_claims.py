"""Claim-verification remediation: the force-correct lines and visible caveats the
job-claim gate emits when a reply describes a job in a way the platform's records
contradict (see _apply_job_claim_gate in agent.py).

This is the declarative remediation TABLE — one _ClaimRemediation per violation `kind`
(state / not_ready / premature_submit / dispatch / reuse) that the detectors in agent.py
produce — kept apart from the (DB-touching) detection so the wording for a new violation
kind is one more table entry, not new control flow.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Honest "wait" guidance injected when the stage-gate fires. Continuation is
# frontend/pull-driven, so a wait only resumes while the page stays open — the
# caveat below states exactly that. Shared with the not_ready advisory in agent.py.
_AWAIT_UPSTREAM_GUIDANCE = (
    "report the true current status to the user and wait — tell them it is still"
    " running and that you'll continue automatically once it finishes, as long as"
    " they keep this page open"
)


@dataclass(frozen=True)
class _ClaimRemediation:
    correction: Callable[[dict[str, Any]], str]  # force-correct line sent back to the model
    caveat: Callable[[dict[str, Any]], str]  # visible caveat appended if it won't correct


def _records_suffix(v: dict[str, Any]) -> str:
    return f" ({v['num']} records)" if v.get("num") is not None else ""


_CLAIM_REMEDIATION: dict[str, _ClaimRemediation] = {
    "state": _ClaimRemediation(
        correction=lambda v: (
            f"- You presented job {v['token']} as finished/ready, but its real"
            f" status is '{v['status']}'. Do NOT claim completion until it is"
            " actually succeeded — state the real status or wait for it."
        ),
        caveat=lambda v: f"job {v['token']} is '{v['status']}', not finished",
    ),
    "not_ready": _ClaimRemediation(
        correction=lambda v: (
            f"- You are advancing the {v['type']} step, but {v['label']}"
            f" ({v['token']}) is still '{v['status']}', not succeeded. Do NOT"
            f" create, confirm, or present a confirmation card for the {v['type']}"
            f" job — it cannot run until that finishes. Instead, {_AWAIT_UPSTREAM_GUIDANCE}."
        ),
        caveat=lambda v: (
            f"{v['label']} ({v['token']}) is still '{v['status']}', so the"
            f" {v['type']} step can't proceed yet"
        ),
    ),
    "premature_submit": _ClaimRemediation(
        correction=lambda v: (
            f"- You described the {v['type']} job as already submitted/queued/running,"
            " but this turn you only rendered a confirmation card — the job is NOT"
            " created yet and has no job ID. It is created only when the user clicks"
            " Confirm on the card. Do NOT claim it is submitted, queued, generating, or"
            " running, and do NOT state a job ID. Tell the user to click Confirm to"
            " start it, then stop and wait."
        ),
        caveat=lambda v: (
            f"the {v['type']} job has not been submitted yet — it starts only when you"
            " click Confirm on the card above"
        ),
    ),
    "dispatch": _ClaimRemediation(
        correction=lambda v: (
            f"- You are about to {'train on' if v['type'] == 'training' else 'evaluate on'}"
            f" dataset {v['token']}{_records_suffix(v)}, but that is an EXISTING dataset"
            " created in an earlier conversation — you did not run a fresh SDG job for this"
            " request this session, nor tell the user you are reusing it. Either submit a"
            " new SDG job for what the user asked for, or explicitly tell the user you are"
            f" REUSING existing dataset {v['token']}{_records_suffix(v)} so they can confirm"
            " it matches before it runs."
        ),
        caveat=lambda v: (
            f"the {v['type']} run is about to use existing dataset {v['token']}"
            f"{_records_suffix(v)} from an earlier conversation, which was not generated or"
            " disclosed as reused this session"
        ),
    ),
    "reuse": _ClaimRemediation(
        correction=lambda v: (
            f"- You implied job {v['token']} was generated/started in this conversation,"
            f" but it is an EXISTING {v['type']} dataset{_records_suffix(v)} created in an"
            " earlier conversation and no matching job was submitted this session. Either"
            " submit a fresh job for the user's request, or explicitly tell the user you"
            f" are REUSING existing job {v['token']}{_records_suffix(v)} so they can confirm"
            " it matches what they asked for."
        ),
        caveat=lambda v: (
            f"job {v['token']} is an existing dataset{_records_suffix(v)} from an earlier"
            " conversation, not a fresh run"
        ),
    ),
}


def _job_claim_correction(violations: list[dict[str, Any]]) -> str:
    lines = [
        "[JOB-CLAIM CHECK — internal system verification, not from the user]",
        "Your reply described job(s) in a way the platform's records contradict:",
    ]
    lines += [_CLAIM_REMEDIATION[v["kind"]].correction(v) for v in violations]
    lines.append("Re-send your reply with the accurate status / reuse disclosure.")
    return "\n".join(lines)


def _job_claim_caveat(violations: list[dict[str, Any]]) -> str:
    parts = [_CLAIM_REMEDIATION[v["kind"]].caveat(v) for v in violations]
    return (
        "\n\n⚠️ Platform records don't match what I said above: "
        + "; ".join(parts)
        + ". Treat these as unconfirmed."
    )
