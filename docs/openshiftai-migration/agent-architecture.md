# Agent Architecture — Research & Recommendation

## Current State

The agent lives in `amortized-ai/amortized` at `src/amortized/agent/`:
- `chat.py` — AsyncOpenAI function-calling loop, SSE streaming, `gpt-5-mini` default
- `tools.py` — 16 tool definitions, each calls the amortized REST API via `httpx` to `localhost:8000`
- `api/agent_routes.py` — `/api/v1/agent/chat` and `/api/v1/agent/chat/stream` endpoints

The studio frontend at `amortized-ai/studio` consumes the SSE stream via `useChat` hook and renders:
- `OptionCards` — clickable multi-choice cards (also auto-parsed from numbered/bulleted lists)
- `ActionCard` — confirm/reject cards for proposed actions
- `PlanProgress` — multi-step plan progress dots
- `ToolBadge` — collapsible tool result indicators

## Who Needs the Agent?

| Client | Has its own agent? | Needs amortized's agent? |
|---|---|---|
| Studio UI | No | **Yes** — only client that needs the chat agent |
| Claude Code (MCP) | Yes — Claude IS the agent | No — calls MCP tools directly |
| CLI | Yes — the human is the agent | No — runs commands directly |

Only the studio UI uses the agent. Claude Code and CLI interact with the amortized API directly.

## Industry Research

### Architecture Patterns (12 platforms surveyed)

| Pattern | Who Does It | LLM Call | Tool Execution | Best For |
|---|---|---|---|---|
| **Embedded endpoint** | Vercel v0, Langfuse | Same server process | Same server (service layer) | Single-team, single-deploy |
| **Separate agent service** | Databricks, Devin, ChatGPT | Dedicated agent server | Agent server calls platform API | Multi-team, independent scaling |
| **Client-side agent** | Cursor, Copilot, Claude Code CLI | Server (proxy) | Client (local filesystem) | IDE/CLI tools needing local access |

### Detailed Findings

**Vercel v0 / AI SDK**
- Agent loop runs server-side as a Next.js route handler
- Tools defined with Zod schemas + `execute()` functions
- SDK auto-loops: LLM call → tool_call → execute → append → re-call
- SSE streaming via "Data Stream Protocol" (typed events)
- `useChat()` hook on client consumes the stream
- No separate service — same deployment

**Cursor / Copilot (IDE agents)**
- LLM call is server-side (Cursor's cloud / GitHub proxy)
- Tool execution is client-side (file edits, terminal on user's machine)
- Multi-model pipeline: main LLM for reasoning, cheap model for applying diffs
- Not applicable to amortized — tools are API operations, not filesystem operations

**ChatGPT / Claude.ai (platform agents)**
- Pure server-side: LLM inference AND tool execution in sandboxed containers
- ChatGPT: ephemeral Docker containers for code execution
- Agent SDK: stream → collect tool calls → execute → append → loop

**Databricks (ML platform agent)**
- Agent as a separate FastAPI service via MLflow AgentServer
- Framework-agnostic wrapper (ResponsesAgent) — can use LangGraph, OpenAI SDK, etc.
- Tools connect via MCP or Unity Catalog functions
- One agent session per deployment, scales independently

**Replit / Devin (full-stack agents)**
- Agent owns the entire environment (VM per session)
- Tightly coupled to the platform — not a reusable pattern

**Prefect Marvin (agent-as-library)**
- Agent is a Python library that runs within user's infra
- Tools are just Prefect tasks
- Lightweight but limited autonomy

### Key Insight

The split is about **where tools execute**, not where the LLM runs (that's always server-side). Amortized's tools are API operations (submit jobs, check status, create datasets) — they execute server-side. This makes the embedded endpoint pattern (Vercel) or separate service pattern (Databricks) the right fit.

## Recommendation

### Phase 1: Clean up the embedded pattern (now)

The current architecture is fundamentally correct. Fix the implementation:

1. **Extract service layer** — Move business logic from route handlers into `services/` modules. Agent tools and API routes both call the same service functions. Kill the HTTP loopback.

2. **Type tool schemas** — Define tools with Pydantic models (Python equivalent of Vercel's Zod schemas). Validate inputs, document parameters, enable auto-generation of tool descriptions.

3. **Add `present_options` tool** — Explicit tool for structured option cards instead of relying on regex parsing of the LLM's text output.

4. **Standardize SSE protocol** — Define typed event types: `delta`, `tool_call`, `tool_result`, `options`, `action`, `plan_progress`, `done`, `error`. Document the protocol so other clients could consume it.

Target architecture:
```
Studio (useChat)  →  SSE  →  Amortized Server (/api/v1/agent/chat/stream)
                                    │
                                    ├── tool: submit_training_job → services/jobs.py
                                    ├── tool: check_job_status   → services/jobs.py
                                    ├── tool: create_dataset     → services/datasets.py
                                    ├── tool: present_options    → SSE options event
                                    ├── tool: propose_action     → SSE action event
                                    └── ... (16 tools total)
```

### Phase 2: Extract to separate service (when needed)

Trigger: when any of these become true:
- Multiple agent personas (studio agent, eval agent, onboarding agent)
- Agent needs independent scaling (many concurrent chat sessions)
- Want to swap agent frameworks (LangGraph, Claude Agent SDK, Pydantic AI agents)
- Different team owns the agent vs the API

At that point, follow the Databricks pattern:
- Agent as its own FastAPI service behind the same API gateway
- Calls amortized REST API (or SDK) for tool execution
- Framework-agnostic wrapper (like Databricks' ResponsesAgent)
- Independent deployment and scaling

### Phase 3: Agent SDK (future)

Build or adopt a thin agent framework:
- Standardize tool definitions, streaming protocol, multi-step loops
- Enable swapping LLM providers without rewriting agent code
- Support multiple agent personas with shared tool definitions
- Candidates: Vercel AI SDK (TypeScript), Pydantic AI (Python), Claude Agent SDK

## References

- Vercel AI SDK 6: https://vercel.com/blog/ai-sdk-6
- Vercel v0 Composite Model: https://vercel.com/blog/v0-composite-model-family
- Vercel AI SDK Agents: https://sdk.vercel.ai/docs/foundations/agents
- Databricks MLflow AgentServer + ResponsesAgent pattern
- Cursor multi-model pipeline architecture
- ChatGPT sandboxed tool execution
- Claude Agent SDK agentic loop
