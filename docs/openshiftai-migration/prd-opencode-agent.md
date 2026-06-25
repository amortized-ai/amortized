# PRD: Replace Amortized Agent with OpenCode

## Problem

Amortized's chat agent is a hand-rolled OpenAI function-calling loop (`agent/chat.py`, 435 lines) with 16 manually defined tools (`agent/tools.py`, 743 lines). Each tool is a hand-written JSON schema + a dispatch function that calls amortized's own HTTP API via httpx on localhost.

This is fragile and redundant:

- **Redundant**: amortized already auto-generates MCP tools from its OpenAPI spec via `fastapi-mcp`. Every API endpoint is already an MCP tool — the 16 hand-written definitions duplicate what MCP provides for free.
- **No planning**: the agent executes one tool at a time with no multi-step reasoning. It can't recover from errors or adapt its approach.
- **No memory**: each chat session starts fresh. The agent can't learn from prior interactions.
- **Manual maintenance**: every new API endpoint requires writing a tool definition, a dispatch handler, and a summary formatter. Miss one and the agent can't use it.
- **Tight coupling**: the agent module imports amortized internals and runs in-process. Scaling, upgrading, or replacing the LLM provider requires changing amortized itself.

## Solution

Replace the in-process agent with **OpenCode** — an open-source coding agent (TypeScript, MIT, github.com/anomalyco/opencode) running as a separate service that connects to amortized via MCP.

### Why OpenCode

- **Built-in HTTP server**: `opencode serve` exposes a REST API on port 4096 with session management and SSE streaming — exactly what Studio needs.
- **Native MCP client**: connects to amortized's existing MCP endpoint. All API tools are discovered automatically.
- **Multi-provider LLM**: supports Anthropic (Claude), OpenAI, Google, and local models. Not locked to one provider.
- **Multi-step reasoning**: plans, executes, retries on error, and adapts — unlike the current single-tool-per-turn loop.
- **Session persistence**: conversations survive across requests.
- **Custom system prompt**: AGENTS.md provides domain knowledge without modifying code.

## Architecture

```
┌──────────────┐    HTTP/SSE     ┌──────────────────┐    MCP (HTTP)    ┌───────────────┐
│    Studio    │ ◄──────────────► │  OpenCode Server │ ◄──────────────► │  amortized    │
│   (React)   │    :4096         │   (TypeScript)   │    :8000/mcp     │  API (:8000)  │
└──────────────┘                 └──────────────────┘                  └───────────────┘
```

**Data flow for a chat message:**

1. User types in Studio chat
2. Studio sends `POST /session/{id}/message` to OpenCode (:4096)
3. OpenCode reasons about the request, decides which tools to call
4. OpenCode calls amortized MCP tools (e.g., `create_job`, `get_job`, `list_flows`)
5. amortized executes the API logic, returns results via MCP
6. OpenCode synthesizes a response and streams it back to Studio via SSE

**Key insight**: amortized's MCP server (`fastapi-mcp`) already exposes every API endpoint as an MCP tool. OpenCode connects to it and gets the full tool set automatically. No hand-written tool definitions needed — ever.

## What Changes

### Removed from Amortized

| File | Reason |
|---|---|
| `src/amortized/agent/chat.py` | Replaced by OpenCode's agent loop |
| `src/amortized/agent/tools.py` | Replaced by auto-generated MCP tools |
| `src/amortized/agent/__init__.py` | Module no longer needed |
| `src/amortized/api/agent_routes.py` | `/api/v1/agent/chat/stream` replaced by OpenCode's API |
| `openai` dependency | Only used by the agent module |

**What stays**: `src/amortized/mcp/server.py` — this is the bridge OpenCode connects to. It stays exactly as-is.

### Added

| Component | Description |
|---|---|
| OpenCode Deployment | K8s Deployment running `opencode serve` on port 4096 |
| OpenCode Service | ClusterIP Service exposing port 4096 |
| AGENTS.md | System prompt with ML domain knowledge (mounted via ConfigMap) |
| opencode.json | MCP server config pointing to amortized's `/mcp` endpoint |
| LLM API key Secret | API key for the agent's reasoning LLM (separate from user keys for SDG) |

### Unchanged

- All amortized API endpoints
- MCP server at `/mcp`
- Job lifecycle, K8s backend, MLflow integration
- Studio's job list, artifact viewer, settings pages
- Everything except the chat interface

## OpenCode Configuration

### AGENTS.md (System Prompt)

The current 180-line system prompt in `chat.py` becomes an AGENTS.md file mounted into the OpenCode container. Content is the same domain knowledge, adapted for OpenCode's format:

```markdown
# Amortized Studio Assistant

You are an AI expert embedded in a web dashboard that helps users build
fine-tuned models to replace expensive frontier model calls.

## Your Role

You handle ALL technical work. The user interacts through a chat UI — they
cannot run commands. Use the amortized MCP tools to interact with the runtime.

## Workflow: SDG → Training → Serve

1. Understand what the user wants (1-2 questions max)
2. Check API keys are configured (list_api_keys tool)
3. Propose an SDG job for data generation
4. After SDG: judge data quality, convert to messages format
5. Propose a training job (LoRA SFT via TRL)
6. After training: check metrics, recommend serving
7. Deploy via training_job_id for automatic model resolution

## asynth Knowledge
[strategy_params, teacher models, input_data, etc.]

## TRL Knowledge
[field names, recommended models, LoRA defaults, etc.]

## Rules
- Take things ONE STEP AT A TIME
- Use sensible defaults — don't ask about lora_r or learning_rate
- Check API keys before proposing SDG jobs
- Keep responses SHORT (3-5 sentences)
```

### opencode.json (MCP Config)

```json
{
  "mcpServers": {
    "amortized": {
      "type": "http",
      "url": "http://amortized-server.amortized.svc.cluster.local:8000/mcp"
    }
  },
  "provider": {
    "type": "anthropic",
    "model": "claude-sonnet-4-20250514"
  }
}
```

The MCP URL uses the K8s service DNS name — OpenCode and amortized are in the same namespace.

## Studio Changes

### Chat API Client

Replace the current SSE client that connects to amortized's `/api/v1/agent/chat/stream`:

```typescript
// Before: direct to amortized agent
const response = await fetch('/api/v1/agent/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ message, history }),
})

// After: via OpenCode session API
// 1. Create or reuse a session
const session = await fetch('http://opencode:4096/session', { method: 'POST' })
const { id: sessionId } = await session.json()

// 2. Send message and stream response
const response = await fetch(`http://opencode:4096/session/${sessionId}/message`, {
  method: 'POST',
  body: JSON.stringify({ content: message }),
})
```

### SSE Event Format

The current agent emits custom SSE events (`thinking`, `tool_result`, `delta`, `action`, `done`). OpenCode has its own streaming format. Studio's `ChatPanel` needs to parse OpenCode's events instead.

Key mapping:

| Current (amortized agent) | OpenCode equivalent |
|---|---|
| `type: "delta"` with `data.text` | Text content chunks in stream |
| `type: "thinking"` with `data.tool` | Tool call events |
| `type: "tool_result"` with `data.summary` | Tool result events |
| `type: "action"` (propose_action) | See below |
| `type: "done"` | Stream completion |

### The propose_action Pattern

Currently, the agent has a special `propose_action` tool that returns a UI button for the user to confirm before submitting a job. This is a custom protocol between the agent and Studio.

With OpenCode, two options:

**Option A (recommended)**: Use the system prompt to instruct OpenCode to emit a specific markdown format when proposing a job:

```markdown
**Proposed: Start Training**
```json
{"type": "training", "config": {"model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", ...}}
```
Click "Confirm" to proceed or tell me to adjust the configuration.
```

Studio parses this markdown pattern and renders a confirmation button. When confirmed, Studio calls the amortized API directly (`POST /api/v1/jobs` with `dry_run: false`), then tells OpenCode "Job submitted: {job_id}".

**Option B**: Create a custom MCP tool `propose_action` on amortized's MCP server that returns a structured response. OpenCode passes this through and Studio renders the button.

Option A is simpler and doesn't require modifying amortized's API.

### Nginx Routing

Studio's nginx config needs a new upstream for OpenCode:

```nginx
upstream opencode {
    server opencode.amortized.svc.cluster.local:4096;
}

location /agent/ {
    proxy_pass http://opencode/;
    proxy_http_version 1.1;
    proxy_set_header Connection '';
    proxy_buffering off;  # SSE
}

location /api/ {
    proxy_pass http://amortized-server:8000/api/;
}
```

Studio hits `/agent/session/...` which proxies to OpenCode. `/api/...` continues to proxy to amortized.

## K8s Manifests

### opencode-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opencode
  namespace: amortized
spec:
  replicas: 1
  selector:
    matchLabels:
      app: opencode
  template:
    metadata:
      labels:
        app: opencode
    spec:
      containers:
      - name: opencode
        image: ghcr.io/anomalyco/opencode:latest
        command: ["opencode", "serve", "--port", "4096"]
        ports:
        - containerPort: 4096
        env:
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: opencode-llm
              key: api-key
        volumeMounts:
        - name: config
          mountPath: /app/.opencode
        resources:
          requests:
            cpu: 200m
            memory: 256Mi
          limits:
            cpu: "1"
            memory: 512Mi
      volumes:
      - name: config
        configMap:
          name: opencode-config
```

### opencode-service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: opencode
  namespace: amortized
spec:
  selector:
    app: opencode
  ports:
  - port: 4096
    targetPort: 4096
```

### opencode-config ConfigMap

Contains:
- `AGENTS.md` — the system prompt
- `opencode.json` — MCP server config

### opencode-llm Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: opencode-llm
  namespace: amortized
type: Opaque
stringData:
  api-key: "sk-ant-..."  # Anthropic API key for agent reasoning
```

This is separate from user API keys (stored encrypted in amortized's DB). This key powers the agent's own LLM calls for reasoning.

## Migration Path

### Phase 1: Deploy OpenCode alongside existing agent

1. Deploy OpenCode on the cluster with MCP config pointing to amortized
2. Verify OpenCode can discover and call amortized's MCP tools
3. Test via OpenCode's TUI or direct API calls
4. Keep the existing agent running — both work in parallel

### Phase 2: Wire Studio to OpenCode

1. Update Studio's chat client to use OpenCode's session API
2. Update SSE parsing for OpenCode's event format
3. Implement the propose_action pattern (Option A or B)
4. Test the full chat flow: Studio → OpenCode → MCP → amortized → response

### Phase 3: Remove old agent

1. Delete `src/amortized/agent/` module
2. Delete `src/amortized/api/agent_routes.py`
3. Remove `openai` from dependencies
4. Update Studio nginx config to remove `/api/v1/agent/` route

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| OpenCode's HTTP server API may change (pre-1.0) | Pin to a specific release tag, not `:latest` |
| OpenCode may not handle amortized's propose_action pattern natively | Use system prompt + markdown parsing (Option A) — no dependency on OpenCode features |
| LLM costs for agent reasoning | Use Claude Sonnet (cheaper) for routine tasks, Opus for complex planning. Configure in opencode.json |
| MCP tool discovery may miss some endpoints | fastapi-mcp generates tools from OpenAPI spec — test coverage by comparing tool count to endpoint count |
| Session state persistence across pod restarts | Mount a PVC for OpenCode's session storage, or treat sessions as ephemeral (acceptable for v1) |

## Success Criteria

1. Studio chat works end-to-end through OpenCode
2. All 16 current agent tools are available via MCP (verified by tool count)
3. SDG → Training → Serve workflow completes via chat
4. Agent can recover from tool errors (e.g., retry a failed job check)
5. No hand-written tool definitions in amortized codebase
6. Adding a new API endpoint to amortized automatically makes it available to the agent

## Verified (2026-06-25)

Local test: OpenCode v1.17.10 → Vertex AI (claude-opus-4-6@default) → MCP → amortized API.

- OpenCode server starts on port 4096, serves web UI + REST API
- Vertex AI provider auto-loads when `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_PROJECT` env vars are set
- MCP remote server connects to `fastapi-mcp` at `http://localhost:8000/mcp` (streamable-http transport)
- Opus 4.6 successfully calls amortized MCP tools (listed all 12 jobs with full details)
- Config format: `.opencode.json` with `mcp.{name}.type: "remote"` (not `mcpServers`, not `type: "http"`)
- Provider config goes via env vars, NOT the `providers` key in config (causes `unrecognized_keys` error)
- MCP may need explicit `POST /mcp` API call to add servers at runtime (config file MCP not always picked up in serve mode)
- Official Docker image: `ghcr.io/anomalyco/opencode:latest` (v1.17.x)
- npm package: `opencode-ai` (not `opencode`)

## Open Questions

1. **Session management**: Should sessions persist across pod restarts (PVC) or be ephemeral? For v1, ephemeral is fine.
2. **Multi-user**: OpenCode's server may not support multiple concurrent users with separate sessions. Need to verify. If not, one OpenCode instance per user (sidecar per Studio pod) or a session multiplexer.
3. **Cost control**: Should there be a token budget per session/user? OpenCode may support this via config.
4. **propose_action UX**: Option A (markdown parsing) vs Option B (custom MCP tool) — recommend starting with Option A for simplicity.
