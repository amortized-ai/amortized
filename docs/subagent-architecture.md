# Subagent Architecture

## Overview

Morty is structured as an **orchestrator + ephemeral subagent** system.
The user sees one continuous conversation — routing between agents is
invisible.

```
Studio → Nginx → Amortized Server (/agent/ proxy) → OpenCode
                                                      ├── morty.md  (orchestrator)
                                                      ├── sdg.md    (subagent)
                                                      └── training.md (subagent)
```

## Conversation Flow

```
User: "Hi"
  → proxy forwards to OpenCode (morty agent)
Morty: "What do you want to do?" [present_options]
User: "Generate training data"
  → orchestrator calls delegate_to_subagent(target="sdg", context="...")
  → proxy intercepts: creates new OpenCode session, sends context with agent="sdg"
  → returns SDG subagent's first response (orchestrator response discarded)
SDG Morty: "What kind of data?" → multiple turns gathering requirements
SDG Morty: job submitted → calls signal_subagent_completion(summary="...")
  → proxy intercepts: discards subagent session, sends summary to orchestrator
  → returns orchestrator's "What's next?" response
User: "Train on this data"
  → new training subagent spins up → same cycle
```

## Components

### 1. Agent Proxy (`src/amortized/api/agent.py`)

Sits between Studio and OpenCode. Endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /agent/session` | Create session → creates OpenCode session, stores mapping |
| `POST /agent/session/{id}/message` | Route message to orchestrator or active subagent |
| `GET /agent/session/{id}/pending` | Proxy pending messages from active session |
| `GET /agent/health` | Health check (proxies to OpenCode) |

**State (in-memory):**

```python
_orchestrator_sessions: dict[str, str]  # external_id → opencode_session_id
_active_subagents: dict[str, str]       # external_id → subagent_session_id
_subagent_targets: dict[str, str]       # external_id → "sdg" or "training"
```

**Routing logic:**

- If active subagent exists → route to it (with `agent=target`)
  - If response contains `signal_subagent_completion` → discard subagent, resume orchestrator
- Else → route to orchestrator
  - If response contains `delegate_to_subagent` → create new session, send context with `agent=target`, return subagent's response

### 2. MCP Signal Tools (`src/amortized/api/ui.py`)

Two thin passthrough endpoints. They just return `{status: "ok"}` — the
proxy scans response parts for these tool calls and acts on them.

| Tool | Called By | Purpose |
|------|-----------|---------|
| `delegate_to_subagent(target, context)` | Orchestrator | Signal proxy to create subagent session |
| `signal_subagent_completion(summary)` | Subagent | Signal proxy to return to orchestrator |

### 3. OpenCode Agents

OpenCode discovers agents from `.opencode/agents/*.md`. The `agent`
field on the message request selects which agent's prompt to use
(`local.agent.set(msg.agent)`). Agent name = filename without `.md`.

| Agent | File | Role |
|-------|------|------|
| `morty` | `agent/prompts/identity.md` + `workflow.md` | Orchestrator — identify intent, delegate, resume, chain |
| `sdg` | `agent/prompts/sdg/workflow.md` | SDG subagent — gather requirements, validate, submit |
| `training` | `agent/prompts/training/workflow.md` | Training subagent — model selection, VRAM, submit |

### 4. Prompt Files

**Orchestrator** (`identity.md` + `workflow.md`):
- Identify intent → present options
- Delegate silently (no text before tool call, never mention subagents)
- Resume after subagent completes → present next steps
- Monitor job events

**Subagent prompts** (`sdg/workflow.md`, `training/workflow.md`):
- Self-contained system prompts with inline identity and conversation rules
- Users address the subagent as "Morty" — they don't know about delegation
- Absorbed content from the old `skills/*/guidance.md` files (routing, model selection, defaults)
- Sub-skill guides (`skills/sdg/classification/guide.md`, etc.) loaded at runtime via Read tool
- Signal completion when job is submitted

### 5. Nginx (`studio/nginx.conf.template`)

Routes `/agent/` to the amortized server backend (which proxies to OpenCode):

```nginx
location ~ ^/agent/(.*)$ {
    proxy_pass http://${BACKEND_HOST}:${BACKEND_PORT}/agent/$1$is_args$args;
}
```

### 6. Config

| Setting | Value | Purpose |
|---------|-------|---------|
| `AMORTIZED_AGENT_UPSTREAM_URL` | `http://opencode:4096` | OpenCode URL for proxy |
| `AMORTIZED_AGENT_SERVER_URL` | `http://amortized-server:8000` | What Studio uses (configmap) |

## Key Design Decisions

**Ephemeral subagents.** Every delegation creates a fresh OpenCode
session. Multiple SDG rounds = multiple independent sessions. No state
accumulation. Orchestrator is the only long-lived session.

**No text before delegation.** The orchestrator prompt says "CRITICAL:
Do NOT write any text before calling delegate_to_subagent." The proxy
replaces the orchestrator's response entirely with the subagent's first
response. The user never sees a gap.

**User message passthrough.** The proxy sends both the orchestrator's
context summary AND the user's original message to the subagent:
```
[CONTEXT]
User wants to generate training data for customer support.

[USER MESSAGE]
I want to build a classifier for support tickets
```

**Tool detection uses `endswith()`.** OpenCode prefixes MCP tool names
differently depending on the server (`amortized_delegate_to_subagent`,
`mcp_amortized__delegate_to_subagent`, etc.). Using `endswith()` is
robust against all prefix variations.

**State is in-memory.** Server restart loses active subagent sessions.
User falls back to orchestrator ("What do you want to do?"). Acceptable
tradeoff — subagent sessions are ephemeral anyway.

## Files Changed (from main)

| File | Change |
|------|--------|
| `agent/prompts/identity.md` | Added orchestrator role |
| `agent/prompts/workflow.md` | Rewritten as router |
| `agent/prompts/sdg/workflow.md` | **New** — SDG subagent prompt |
| `agent/prompts/training/workflow.md` | **New** — Training subagent prompt |
| `agent/skills/sdg/guidance.md` | **Deleted** — absorbed into sdg/workflow.md |
| `agent/skills/training/guidance.md` | **Deleted** — absorbed into training/workflow.md |
| `src/amortized/api/agent.py` | **New** — session proxy with routing |
| `src/amortized/api/ui.py` | Added delegation MCP tools, improved present_options |
| `src/amortized/config.py` | Added `agent_upstream_url` |
| `src/amortized/main.py` | Mounted agent router |
| `k8s/base/opencode-*.yaml` | **New** — OpenCode deployment with subagent mounts |
| `k8s/base/kustomization.yaml` | Added OpenCode resources + configMapGenerators |
| `k8s/base/configmap.yaml` | Set agent server URL |
| `studio/nginx.conf.template` | Route /agent/ to backend |
| `Makefile` | Updated prompt target |

## Known Gaps

1. **Event routing** — `[SYSTEM EVENT]` for job status changes should
   route through the proxy to the active session. Currently the event
   push mechanism bypasses the proxy.

2. **Session persistence** — In-memory state. Server restart = all
   active subagent sessions lost. Could persist to DB if needed.

3. **Deploy repo sync** — `amortized-deploy` repo's `base-internal/`
   has OpenCode resources that now conflict with the ones in this repo.
   The `base-internal/kustomization.yaml` needs updating to remove
   duplicates.
