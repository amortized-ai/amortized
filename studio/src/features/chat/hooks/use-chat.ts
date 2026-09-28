import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { sendOpenCodeMessage, fetchSessionMessages, fetchPendingMessages, generateChatTitle, createJob, getBaseUrl } from "@/lib/api-client"
import { useChatStore, type PersistedMessage } from "@/stores/chat-store"
import { useSettingsStore } from "@/stores/settings-store"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-chat")
import type {
  ChatMessage,
  ChatState,
  ProposedAction,
  ToolResult,
  OpenCodeResponse,
} from "../types"
import { extractValidatedJobConfig, VALIDATE_TO_CREATE_ENDPOINT } from "../utils/parse-tool-result"

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

// Conversations with a send in flight. Module-level so it survives the keyed
// unmount/remount that happens when navigating between conversations mid-stream:
// the send closure keeps writing to the store after its component unmounts, and the
// remounted hook reads this Set to know it should still show "streaming".
const _activeRequests = new Set<string>()

// Guards concurrent sends to the SAME conversation (double-submit). Keyed by convId and
// always released under that same key (see sendMessage) so it can never leak.
const _sendLock = new Set<string>()

// Map a persisted message (the single source of truth) to the ChatMessage the UI renders.
function toChatMessage(m: PersistedMessage): ChatMessage {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    timestamp: m.timestamp,
    toolResults: m.toolResults ?? [],
    proposedAction: m.proposedAction ?? null,
    optionCards: m.optionCards ?? [],
    selectedOptionValue: m.selectedOptionValue,
    phase: m.phase,
    streamStartedAt: m.streamStartedAt,
  }
}

const TOOL_BLOCK_RE =
  /<function_(?:calls|details|response|returns)>[\s\S]*?<\/function_(?:calls|details|response|returns)>/g
const INVOKE_NAME_RE = /<invoke name="([^"]+)">/g
const STRAY_TAG_RE = /<\/?(?:invoke|parameter|summary)[^>]*>[^<]*/g

function extractToolCalls(text: string): {
  cleanText: string
  tools: ToolResult[]
} {
  const seen = new Set<string>()
  for (const m of text.matchAll(INVOKE_NAME_RE)) {
    seen.add(normalizeToolName(m[1]!))
  }
  const tools: ToolResult[] = [...seen].map((name) => ({
    name,
    result: "",
    collapsed: true,
  }))
  const cleaned = text
    .replace(TOOL_BLOCK_RE, "")
    .replace(STRAY_TAG_RE, "")
    .replace(/<phase>[\w:_-]+<\/phase>/g, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
  return { cleanText: cleaned, tools }
}

function parseOpenCodeResponse(response: OpenCodeResponse): {
  content: string
  toolResults: ToolResult[]
} {
  let rawText = ""
  const toolResults: ToolResult[] = []

  for (const part of response.parts) {
    if (part.type === "text") {
      rawText += part.text ?? ""
    } else if (part.type === "tool") {
      toolResults.push({
        name: normalizeToolName(part.tool),
        result: serializeToolOutput(part.output ?? part.state ?? ""),
        collapsed: true,
      })
    }
  }

  const { cleanText, tools } = extractToolCalls(rawText)
  const seen = new Set(toolResults.map((t) => t.name.toLowerCase()))
  for (const t of tools) {
    const key = t.name.toLowerCase()
    if (!seen.has(key)) {
      seen.add(key)
      toolResults.push(t)
    }
  }

  return { content: cleanText, toolResults }
}

const UI_TOOLS = new Set([
  "present_options",
  "signal_phase",
  "get_model_pricing",
  "show_model_pricing",
  "estimate_training_resources",
  "show_vram_estimate",
  "submit_recipe_job",
  "create_sdg_job",
  "create_training_job",
  "validate_sdg_job",
  "validate_training_job",
  "validate_eval_job",
  "validate_recipe_job",
  "create_job",
  "split_dataset",
])

const ALL_TURN_TOOLS = new Set(["signal_phase"])

const JOB_CREATION_TOOLS = new Set([
  "create_sdg_job",
  "create_training_job",
  "create_eval_job",
  "submit_recipe_job",
  "create_job",
  "split_dataset",
])

function normalizeToolName(raw: string): string {
  return raw.replace(/^(?:mcp_amortized__|amortized_)/, "")
}

function serializeToolOutput(value: unknown): string {
  if (typeof value === "string") return value
  return JSON.stringify(value)
}

function extractSessionData(
  sessionMessages: OpenCodeResponse[],
  existingTools: ToolResult[],
): { tools: ToolResult[]; text: string } {
  const seen = new Set(existingTools.map((t) => t.name.toLowerCase()))
  const tools = [...existingTools]
  const textParts: string[] = []

  let lastUserIdx = -1
  for (let i = sessionMessages.length - 1; i >= 0; i--) {
    const info = (sessionMessages[i] as unknown as Record<string, unknown>).info as Record<string, unknown> | undefined
    if (info?.role === "user") { lastUserIdx = i; break }
  }
  const currentTurnMessages = lastUserIdx >= 0
    ? sessionMessages.slice(lastUserIdx + 1)
    : sessionMessages

  for (const msg of sessionMessages) {
    const info = (msg as unknown as Record<string, unknown>).info as Record<string, unknown> | undefined
    if (info?.role !== "assistant") continue

    for (const part of msg.parts) {
      if (part.type === "tool") {
        const name = normalizeToolName(part.tool ?? "")
        if (ALL_TURN_TOOLS.has(name)) {
          const stateObj = part.state as Record<string, unknown> | undefined
          const rawOutput = part.output ?? stateObj?.output ?? ""
          const output = typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput)
          tools.push({ name, result: output, collapsed: true })
        }
      }
    }
  }

  for (const msg of currentTurnMessages) {
    const info = (msg as unknown as Record<string, unknown>).info as Record<string, unknown> | undefined
    if (info?.role !== "assistant") continue

    for (const part of msg.parts) {
      if (part.type === "text" && part.text) {
        textParts.push(part.text)
      } else if (part.type === "tool") {
        const name = normalizeToolName(part.tool ?? "")
        // Validate tools legitimately fire multiple times in one turn —
        // the agent fixes a validation error and re-validates. Dropping
        // the retry (the default dedup) keeps only the ERRORED result,
        // which kills the confirmation card. The card builder below picks
        // the last result that actually parses as a valid config.
        const allowDuplicates = name === "create_sdg_job" || name === "create_training_job" || name === "submit_recipe_job" || name === "create_job" || name === "split_dataset" || name.startsWith("validate_")
        if (UI_TOOLS.has(name) && !ALL_TURN_TOOLS.has(name) && (allowDuplicates || !seen.has(name.toLowerCase()))) {
          if (!allowDuplicates) seen.add(name.toLowerCase())
          const stateObj = part.state as Record<string, unknown> | undefined
          const rawOutput = part.output ?? stateObj?.output ?? ""
          const output = typeof rawOutput === "string" ? rawOutput : JSON.stringify(rawOutput)
          tools.push({ name, result: output, collapsed: true })
        }
      }
    }
  }

  const { cleanText } = extractToolCalls(textParts.join("\n"))
  return { tools, text: cleanText }
}

// Build the job-confirmation card from a turn's tool results. The agent may
// validate several times in one turn (fix an error, re-validate), so walk
// backwards and use the most recent result that parses as a valid config — an
// earlier ERRORED result must not shadow a later successful one. Shared by every
// path that renders an assistant turn (user send, async poll, job-finished
// notify) so the card is never dropped depending on how the turn was triggered.
function buildProposedAction(toolResults: ToolResult[]): ProposedAction | null {
  for (let i = toolResults.length - 1; i >= 0; i--) {
    const t = toolResults[i]!
    if (!(t.name in VALIDATE_TO_CREATE_ENDPOINT)) continue
    const validated = extractValidatedJobConfig(t.result)
    if (validated) {
      return {
        action: `Create ${validated.jobType.toUpperCase()} Job`,
        description: `Submit this ${validated.jobType} job?`,
        params: validated.config,
        jobType: validated.jobType as "sdg" | "training" | "eval" | "serve",
        endpoint: VALIDATE_TO_CREATE_ENDPOINT[t.name],
        config: validated.config,
        parentJobId: validated.parentJobId,
        recipe: validated.recipe,
      }
    }
  }
  return null
}

// Derive the current phase label (`phase` or `phase:step`) from a turn's signal_phase
// tool result. Extracted so the four render paths (send, async poll, job-notify,
// mount reconcile) stay in sync.
function extractPhase(toolResults: ToolResult[]): string | undefined {
  const phaseTool = [...toolResults].reverse().find((t) => t.name === "signal_phase")
  if (!phaseTool?.result) return undefined
  try {
    const p = typeof phaseTool.result === "string" ? JSON.parse(phaseTool.result) : phaseTool.result
    if (p?.phase) return p.step ? `${p.phase}:${p.step}` : p.phase
  } catch { /* ignore */ }
  return undefined
}

function startThinkingTimer(
  convId: string,
  setStep: (s: string | null) => void,
  startTime?: number,
): () => void {
  const start = startTime ?? Date.now()
  let acting = false
  let pollDone = false

  const elapsed = (Date.now() - start) / 1000
  if (elapsed > 15) {
    setStep("Still working...")
  } else {
    setStep("Thinking...")
  }

  const timerId = setInterval(() => {
    const elapsed = (Date.now() - start) / 1000
    if (elapsed > 15) {
      setStep("Still working...")
    } else if (acting) {
      setStep("Acting...")
    }
  }, 1000)

  const pollId = setInterval(async () => {
    if (pollDone || acting) return
    try {
      const msgs = await fetchSessionMessages(convId)
      if (pollDone) return
      for (const msg of msgs) {
        if (msg.parts?.some((p: { type: string }) => p.type === "tool")) {
          acting = true
          setStep("Acting...")
          break
        }
      }
    } catch { /* ignore */ }
  }, 3000)

  return () => {
    pollDone = true
    clearInterval(timerId)
    clearInterval(pollId)
    setStep(null)
  }
}

/**
 * Chat hook. `messages` is derived directly from the zustand store (the single source
 * of truth), so unmounting mid-stream loses nothing: the in-flight send closure keeps
 * writing to the store and the remounted hook re-derives from it. Mounted inside a
 * component keyed by conversation id, so React handles conversation switching via
 * unmount/remount.
 */
export function useChat() {
  const currentConversationId = useChatStore((s) => s.currentConversationId)
  const setCurrentConversationId = useChatStore((s) => s.setCurrentConversationId)
  const addConversation = useChatStore((s) => s.addConversation)
  const addMessage = useChatStore((s) => s.addMessage)
  const updateConversationTitle = useChatStore((s) => s.updateConversationTitle)

  // Single source of truth: select the raw persisted array (a stable reference that only
  // changes when THIS conversation's messages change), then map to ChatMessage. No local
  // copy, no manual store<->local sync.
  const persisted = useChatStore((s) => {
    const conv = s.conversations.find((c) => c.id === s.currentConversationId)
    return conv?.messages
  })
  const messages = useMemo<ChatMessage[]>(
    () => (persisted ? persisted.map(toChatMessage) : []),
    [persisted],
  )

  const [error, setError] = useState<string | null>(null)
  const [thinkingStep, setThinkingStep] = useState<string | null>(null)
  const stopThinkingRef = useRef<(() => void) | null>(null)

  // chatState is per-mount UI state. On mount, if a send for this conversation is still in
  // flight (the module-level Set survives the keyed remount), start in "streaming".
  const [chatState, setChatState] = useState<ChatState>(() =>
    currentConversationId && _activeRequests.has(currentConversationId) ? "streaming" : "idle",
  )
  const [currentToolCall, setCurrentToolCall] = useState<ToolResult | null>(null)

  const chatStateRef = useRef(chatState)
  useEffect(() => { chatStateRef.current = chatState }, [chatState])

  // Always stop any thinking timer when this hook unmounts, so a send whose component
  // unmounted mid-stream can't leave the 1s/3s intervals running.
  useEffect(() => () => {
    stopThinkingRef.current?.()
    stopThinkingRef.current = null
  }, [])

  const warmupPromiseRef = useRef<Promise<void> | null>(null)

  // On mount, reconcile a conversation that was mid-stream.
  useEffect(() => {
    const convId = currentConversationId
    if (!convId) return

    // Case 1 — a send is still in flight (navigated away and back within the app). The
    // detached send closure keeps writing to the store, so messages are already correct;
    // we only mirror "streaming" until the request clears, then settle chatState from the
    // final stored turn. Both the timer and the poll are cleaned up on unmount.
    if (_activeRequests.has(convId)) {
      const stored = useChatStore.getState().getConversationMessages(convId)
      const last = stored[stored.length - 1]
      stopThinkingRef.current = startThinkingTimer(convId, setThinkingStep, last?.streamStartedAt)
      let cancelled = false
      const pollId = setInterval(() => {
        if (cancelled) return
        if (_activeRequests.has(convId)) return
        clearInterval(pollId)
        stopThinkingRef.current?.()
        stopThinkingRef.current = null
        const msgs = useChatStore.getState().getConversationMessages(convId)
        const lastMsg = msgs[msgs.length - 1]
        const action = lastMsg ? buildProposedAction(lastMsg.toolResults ?? []) : null
        setChatState(action ? "action_pending" : "done")
      }, 500)
      return () => {
        cancelled = true
        clearInterval(pollId)
        stopThinkingRef.current?.()
        stopThinkingRef.current = null
      }
    }

    // Case 2 — no in-flight send, but the last turn is an empty assistant placeholder with
    // no tool output: an orphan from a send whose in-memory continuation was lost (a full
    // page reload). Mark it interrupted so the turn reads honestly and the user can resend.
    // A completed tool-only turn has toolResults, so it is left untouched.
    const stored = useChatStore.getState().getConversationMessages(convId)
    const last = stored[stored.length - 1]
    if (last && last.role === "assistant" && !last.content && (last.toolResults?.length ?? 0) === 0) {
      useChatStore.getState().updateMessageFields(convId, last.id, {
        content: "Response was interrupted. Send a new message to continue.",
      })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only reconciliation
  }, [])

  // Session warmup: verify the OpenCode session is still live, else rebuild it and replay
  // context. Recorded in a ref so a send awaits it before firing.
  useEffect(() => {
    if (!currentConversationId) return
    const sessionId = useChatStore.getState().getSessionId(currentConversationId)
    if (!sessionId) return
    const msgs = useChatStore.getState().getConversationMessages(currentConversationId)
    if (msgs.length === 0) return

    const convId = currentConversationId
    let cancelled = false

    async function warmup() {
      try {
        const resp = await fetch(`${getBaseUrl()}/agent/session/${sessionId}/message`)
        if (resp.ok) {
          if (!cancelled) useChatStore.getState().setSessionStatus(convId, "connected")
          return
        }
      } catch { /* session is stale or unreachable */ }

      if (cancelled) return
      useChatStore.getState().setSessionStatus(convId, "reconnecting")
      useChatStore.getState().clearSessionId(convId)

      try {
        const createResp = await fetch(`${getBaseUrl()}/agent/session`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        })
        if (!createResp.ok) throw new Error("Failed to create session")
        const { id: newSessionId } = await createResp.json()
        useChatStore.getState().setSessionId(convId, newSessionId)

        const { summarizeConversation } = await import("@/lib/context-summarizer")
        const summary = summarizeConversation(msgs)
        if (summary) {
          const replayResp = await fetch(`${getBaseUrl()}/agent/session/${newSessionId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ agent: "morty", parts: [{ type: "text", text: summary }] }),
          })
          if (!replayResp.ok) {
            logger.warn("session context replay failed", { convId, status: replayResp.status })
          }
        }
        if (!cancelled) {
          useChatStore.getState().setSessionStatus(convId, "restored")
          logger.info("session warmup complete", { convId, newSessionId })
        }
      } catch (err) {
        if (!cancelled) {
          useChatStore.getState().clearSessionId(convId)
          useChatStore.getState().setSessionStatus(convId, "unknown")
          logger.warn("session warmup failed", { convId, error: err instanceof Error ? err.message : String(err) })
        }
      }
    }

    warmupPromiseRef.current = warmup()
    return () => { cancelled = true }
  }, [currentConversationId])

  // Poll for asynchronous follow-up turns the agent produced out of band, appending them
  // to the store (the derived selector renders them).
  useEffect(() => {
    if (!currentConversationId) return
    const convId = currentConversationId

    const pollPending = async () => {
      if (chatStateRef.current === "streaming") return

      try {
        const pending = await fetchPendingMessages(convId)
        if (pending.length === 0) return

        for (const response of pending) {
          const parsed = parseOpenCodeResponse(response)
          const session = extractSessionData([response], parsed.toolResults)
          const proposedAction = buildProposedAction(session.tools)

          addMessage(convId, {
            id: generateId(),
            role: "assistant",
            content: session.text || parsed.content,
            timestamp: new Date().toISOString(),
            toolResults: session.tools,
            proposedAction,
            phase: extractPhase(session.tools),
          })
        }
      } catch {
        // Silently continue polling
      }
    }

    const interval = setInterval(pollPending, 5000)
    return () => clearInterval(interval)
  }, [currentConversationId, addMessage])

  const clearProposedActions = useCallback(() => {
    const convId = currentConversationId
    if (!convId) return
    const store = useChatStore.getState()
    for (const m of store.getConversationMessages(convId)) {
      if (m.proposedAction) {
        store.updateMessageFields(convId, m.id, { proposedAction: null })
      }
    }
  }, [currentConversationId])

  const selectOption = useCallback((messageId: string, value: string) => {
    if (currentConversationId) {
      useChatStore.getState().updateMessageFields(currentConversationId, messageId, { selectedOptionValue: value })
    }
  }, [currentConversationId])

  const sendMessage = useCallback(
    async (content: string) => {
      // Resolve the conversation id synchronously up front (generate one for a new
      // conversation) so the lock is keyed consistently and acquired BEFORE any await —
      // no check-then-act gap, and the same key is released in `finally`.
      const existingConvId = currentConversationId ?? useChatStore.getState().currentConversationId
      const convId = existingConvId ?? `conv-${Date.now()}`
      const isNewConversation = !existingConvId

      if (chatState === "streaming" || _sendLock.has(convId)) return
      _sendLock.add(convId)

      const assistantId = generateId()
      try {
        setError(null)
        setCurrentToolCall(null)
        if (chatState === "action_pending") clearProposedActions()

        let needsAutoTitle = false
        if (isNewConversation) {
          addConversation({
            id: convId,
            title: "New conversation",
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            messages: [],
          })
          needsAutoTitle = true
        } else {
          const conv = useChatStore.getState().conversations.find((c) => c.id === convId)
          const userMessageCount = conv?.messages.filter((m) => m.role === "user").length || 0
          if (conv && conv.title === "New conversation" && userMessageCount === 0) {
            needsAutoTitle = true
          }
        }

        // Count history BEFORE this turn's messages so the "rebuilt" status below can tell a
        // real session rebuild (existing history, no live session) from a new conversation's
        // first send (no history) — the latter must not be flagged as rebuilt.
        const priorMessageCount = useChatStore.getState().getConversationMessages(convId).length

        const streamStart = Date.now()
        addMessage(convId, {
          id: generateId(),
          role: "user",
          content,
          timestamp: new Date().toISOString(),
        })
        addMessage(convId, {
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: new Date().toISOString(),
          streamStartedAt: streamStart,
        })
        _activeRequests.add(convId)

        // Make this conversation current now so the keyed view shows the new turn
        // immediately (the messages selector keys off currentConversationId).
        if (isNewConversation) setCurrentConversationId(convId)

        setChatState("streaming")
        stopThinkingRef.current = startThinkingTimer(convId, setThinkingStep, streamStart)

        if (warmupPromiseRef.current) {
          await warmupPromiseRef.current
          warmupPromiseRef.current = null
        }

        try {
          const hadPriorSession = !!useChatStore.getState().getSessionId(convId)
          logger.info("sending to OpenCode", { conversationId: convId })
          const { chatModelSelection } = useSettingsStore.getState()
          const response = await sendOpenCodeMessage(convId, content, chatModelSelection)
          stopThinkingRef.current?.()
          stopThinkingRef.current = null
          logger.info("OpenCode response received", {
            provider: response.info?.providerID,
            model: response.info?.modelID,
            parts: response.parts?.length,
          })

          const parsed = parseOpenCodeResponse(response)
          const sessionMessages = await fetchSessionMessages(convId)
          const session = extractSessionData(sessionMessages, parsed.toolResults)
          const toolResults = session.tools
          const responseContent = session.text || parsed.content

          if (toolResults.length > 0) {
            setCurrentToolCall(toolResults[toolResults.length - 1]!)
          }

          const proposedAction = buildProposedAction(toolResults)

          useChatStore.getState().updateMessageFields(convId, assistantId, {
            content: responseContent,
            toolResults,
            proposedAction,
            phase: extractPhase(toolResults),
          })

          _activeRequests.delete(convId)

          if (toolResults.some((t) => JOB_CREATION_TOOLS.has(t.name))) {
            useChatStore.getState().setJobInFlight(convId, true)
          }

          setChatState(proposedAction ? "action_pending" : "done")
          useChatStore.getState().setSessionStatus(convId, "connected")

          if (!hadPriorSession && priorMessageCount > 0) {
            useChatStore.getState().setSessionStatus(convId, "rebuilt")
          }

          if (needsAutoTitle) {
            generateChatTitle(content).then((title) => {
              updateConversationTitle(convId, title)
            }).catch(() => {
              updateConversationTitle(convId, content.slice(0, 40) + (content.length > 40 ? "..." : ""))
            })
          }
        } catch (err) {
          stopThinkingRef.current?.()
          stopThinkingRef.current = null
          _activeRequests.delete(convId)
          logger.error("OpenCode error", { error: err instanceof Error ? err.message : String(err) })
          // Keep the assistant placeholder and persist the error into it, so the failed
          // turn survives navigation instead of vanishing.
          useChatStore.getState().updateMessageFields(convId, assistantId, {
            content: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
          })
          setError(err instanceof Error ? err.message : "Unknown error")
          setChatState("error")
        }
      } finally {
        _sendLock.delete(convId)
      }
    },
    [
      chatState,
      currentConversationId,
      addConversation,
      addMessage,
      updateConversationTitle,
      setCurrentConversationId,
      clearProposedActions,
    ],
  )

  const confirmAction = useCallback(async () => {
    const convId = currentConversationId ?? useChatStore.getState().currentConversationId
    if (!convId) return
    const msgs = useChatStore.getState().getConversationMessages(convId)
    const actionMsg = [...msgs].reverse().find((m) => m.proposedAction)
    if (!actionMsg?.proposedAction) return

    const { endpoint, config, parentJobId, recipe, jobType } = actionMsg.proposedAction as ProposedAction
    if (!endpoint) return

    let body: Record<string, unknown>
    if (endpoint === "/api/v1/jobs/recipe") {
      body = { recipe, overrides: {}, parent_job_id: parentJobId }
    } else {
      body = { ...config }
      if (parentJobId) body.parent_job_id = parentJobId
    }

    setChatState("streaming")

    try {
      const job = await createJob(endpoint, body)

      const jobToolResult: ToolResult = {
        name: `create_${jobType}_job`,
        result: JSON.stringify(job),
        collapsed: true,
      }

      const store = useChatStore.getState()
      store.setJobInFlight(convId, true)
      store.updateMessageFields(convId, actionMsg.id, {
        proposedAction: null,
        toolResults: [...(actionMsg.toolResults ?? []), jobToolResult],
      })

      const sessionId = store.getSessionId(convId)
      if (sessionId) {
        try {
          const resp = await fetch(`${getBaseUrl()}/agent/session/${sessionId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              agent: "morty",
              parts: [{ type: "text", text: `Job confirmed and submitted. Job ID: ${job.id} (${jobType} job, status: ${job.status})` }],
            }),
          })
          if (!resp.ok) {
            logger.warn("morty notification failed", { status: resp.status })
          }
        } catch (notifyErr) {
          logger.warn("morty notification error", { error: notifyErr instanceof Error ? notifyErr.message : String(notifyErr) })
        }
      }

      setChatState("done")
    } catch (err) {
      useChatStore.getState().setJobInFlight(convId, false)
      setError(err instanceof Error ? err.message : "Failed to submit job")
      setChatState("error")
    }
  }, [currentConversationId])

  const rejectAction = useCallback(async () => {
    const convId = currentConversationId ?? useChatStore.getState().currentConversationId
    if (!convId) return
    const hasAction = useChatStore.getState().getConversationMessages(convId).some((m) => m.proposedAction)
    if (!hasAction) return

    clearProposedActions()

    const sessionId = useChatStore.getState().getSessionId(convId)
    if (sessionId) {
      try {
        const resp = await fetch(`${getBaseUrl()}/agent/session/${sessionId}/message`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            agent: "morty",
            parts: [{ type: "text", text: "Job submission was cancelled by the user. Ask what they'd like to change." }],
          }),
        })
        if (!resp.ok) {
          logger.warn("morty rejection notification failed", { status: resp.status })
        }
      } catch (notifyErr) {
        logger.warn("morty rejection notification error", { error: notifyErr instanceof Error ? notifyErr.message : String(notifyErr) })
      }
    }

    setChatState("done")
  }, [currentConversationId, clearProposedActions])

  const jobNotifyQueueRef = useRef<Array<{ jobId: string; jobType: string; status: string; convId: string }>>([])
  const jobNotifyRunningRef = useRef(false)

  const processJobNotifyQueue = useCallback(async () => {
    if (jobNotifyRunningRef.current) return
    jobNotifyRunningRef.current = true

    const BLOCKED_STATES = new Set<ChatState>(["streaming", "tool_call", "action_pending"])
    const MAX_WAIT_MS = 60_000
    const MAX_FAILURES = 3
    let consecutiveFailures = 0
    let waitedMs = 0

    while (jobNotifyQueueRef.current.length > 0) {
      const nextConvId = jobNotifyQueueRef.current[0]!.convId
      if (_sendLock.has(nextConvId) || BLOCKED_STATES.has(chatStateRef.current)) {
        if (waitedMs >= MAX_WAIT_MS) {
          logger.warn("job notify queue timed out waiting for idle state")
          jobNotifyQueueRef.current.length = 0
          break
        }
        await new Promise((r) => setTimeout(r, 500))
        waitedMs += 500
        continue
      }
      waitedMs = 0

      if (consecutiveFailures >= MAX_FAILURES) {
        logger.warn("job notify queue circuit breaker tripped", { failures: consecutiveFailures })
        jobNotifyQueueRef.current.length = 0
        break
      }

      const { jobId, jobType, status, convId } = jobNotifyQueueRef.current.shift()!
      useChatStore.getState().setJobInFlight(convId, false)

      const placeholderId = generateId()
      _sendLock.add(convId)
      setChatState("streaming")

      addMessage(convId, {
        id: placeholderId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
      })
      _activeRequests.add(convId)

      try {
        const response = await sendOpenCodeMessage(
          convId,
          `Job ${jobId} (${jobType}) finished with status: ${status}. Use present_options to suggest next steps to the user.`,
        )

        const parsed = parseOpenCodeResponse(response)
        const sessionMessages = await fetchSessionMessages(convId)
        const session = extractSessionData(sessionMessages, parsed.toolResults)
        // A job-finished turn may self-heal and re-validate (e.g. a failed job
        // whose config the agent then fixes), so it can carry a confirmation card.
        const proposedAction = buildProposedAction(session.tools)

        useChatStore.getState().updateMessageFields(convId, placeholderId, {
          content: session.text || parsed.content,
          toolResults: session.tools,
          proposedAction,
          phase: extractPhase(session.tools),
        })

        _activeRequests.delete(convId)
        useChatStore.getState().addNotifiedJob(convId, jobId)
        setChatState(proposedAction ? "action_pending" : "done")
        consecutiveFailures = 0
      } catch (err) {
        _activeRequests.delete(convId)
        useChatStore.getState().updateMessageFields(convId, placeholderId, {
          content: "I wasn't able to suggest next steps. You can ask me what to do next.",
        })
        setChatState("done")
        consecutiveFailures++
        logger.error("job completion notification failed", { jobId, consecutiveFailures, error: err instanceof Error ? err.message : String(err) })
      } finally {
        _sendLock.delete(convId)
      }
    }

    jobNotifyRunningRef.current = false
  }, [addMessage])

  const notifyJobComplete = useCallback(async (jobId: string, jobType: string, status: string) => {
    const convId = currentConversationId ?? useChatStore.getState().currentConversationId
    if (!convId) return

    const notified = useChatStore.getState().getNotifiedJobs(convId)
    if (notified.includes(jobId)) return

    jobNotifyQueueRef.current.push({ jobId, jobType, status, convId })
    await processJobNotifyQueue()
  }, [currentConversationId, processJobNotifyQueue])

  const jobInFlight = useChatStore(
    (s) => s.jobInFlight[s.currentConversationId ?? ""] ?? false,
  )

  useEffect(() => {
    if (!jobInFlight || !currentConversationId) return
    const timeout = setTimeout(() => {
      logger.warn("jobInFlight auto-expired after timeout", { convId: currentConversationId })
      useChatStore.getState().setJobInFlight(currentConversationId, false)
    }, 30 * 60 * 1000)
    return () => clearTimeout(timeout)
  }, [jobInFlight, currentConversationId])

  const isStreaming = chatState === "streaming" || chatState === "tool_call" || jobInFlight

  const latestAction = [...messages].reverse().find((m) => m.proposedAction !== null)?.proposedAction ?? null

  return {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    error,
    chatState,
    currentToolCall,
    thinkingStep,
    proposedAction: latestAction,
    confirmAction,
    rejectAction,
    notifyJobComplete,
  }
}
