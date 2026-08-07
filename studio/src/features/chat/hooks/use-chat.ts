import { useCallback, useEffect, useRef, useState } from "react"
import { sendOpenCodeMessage, fetchSessionMessages, fetchPendingMessages, generateChatTitle, createJob } from "@/lib/api-client"
import { useChatStore } from "@/stores/chat-store"
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

function restoreMessages(
  getConversationMessages: (id: string) => import("@/stores/chat-store").PersistedMessage[],
  conversationId: string | null,
): ChatMessage[] {
  if (!conversationId) return []
  const persisted = getConversationMessages(conversationId)
  return persisted.map((m) => ({
    id: m.id,
    role: m.role,
    content: m.content,
    timestamp: m.timestamp,
    toolResults: m.toolResults ?? [],
    proposedAction: m.proposedAction ?? null,
    optionCards: m.optionCards ?? [],
    selectedOptionValue: m.selectedOptionValue,
    phase: m.phase,
  }))
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
  "validate_recipe_job",
])

const ALL_TURN_TOOLS = new Set(["signal_phase"])

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
        const allowDuplicates = name === "create_sdg_job" || name === "create_training_job" || name === "submit_recipe_job"
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

const _firedJobEvents = new Set<string>()

/**
 * Chat hook. Designed to be mounted inside a keyed component so that
 * React handles conversation switching via unmount/remount — no
 * effect-based message restoration needed.
 */
export function useChat() {
  const {
    currentConversationId,
    setCurrentConversationId,
    addConversation,
    addMessage,
    updateConversationTitle,
    getConversationMessages,
  } = useChatStore()

  const [chatState, setChatState] = useState<ChatState>("idle")
  const [error, setError] = useState<string | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    return restoreMessages(getConversationMessages, currentConversationId)
  })

  const messagesRef = useRef(messages)
  useEffect(() => { messagesRef.current = messages }, [messages])

  const [currentToolCall, setCurrentToolCall] = useState<ToolResult | null>(null)

  const chatStateRef = useRef(chatState)
  useEffect(() => { chatStateRef.current = chatState }, [chatState])

  const warmupPromiseRef = useRef<Promise<void> | null>(null)

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
        const resp = await fetch(`/agent/session/${sessionId}/message`)
        if (resp.ok) {
          if (!cancelled) useChatStore.getState().setSessionStatus(convId, "connected")
          return
        }
      } catch { /* session is stale or unreachable */ }

      if (cancelled) return
      useChatStore.getState().setSessionStatus(convId, "reconnecting")
      useChatStore.getState().clearSessionId(convId)

      try {
        const createResp = await fetch("/agent/session", {
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
          await fetch(`/agent/session/${newSessionId}/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ agent: "morty", parts: [{ type: "text", text: summary }] }),
          })
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

          const phaseTool = session.tools.find((t) => t.name === "signal_phase")
          let phase: string | null = null
          if (phaseTool?.result) {
            try {
              const p = typeof phaseTool.result === "string" ? JSON.parse(phaseTool.result) : phaseTool.result
              if (p?.phase) phase = p.step ? `${p.phase}:${p.step}` : p.phase
            } catch { /* ignore */ }
          }

          const msgId = generateId()
          const followUp: ChatMessage = {
            id: msgId,
            role: "assistant",
            content: session.text || parsed.content,
            timestamp: new Date().toISOString(),
            toolResults: session.tools,
            proposedAction: null,
            optionCards: [],
            phase: phase ?? undefined,
          }

          setMessages((prev) => [...prev, followUp])
          addMessage(convId, {
            id: msgId,
            role: "assistant",
            content: followUp.content,
            timestamp: followUp.timestamp,
            toolResults: followUp.toolResults,
            phase: followUp.phase,
          })
        }
      } catch {
        // Silently continue polling
      }
    }

    const interval = setInterval(pollPending, 5000)
    return () => clearInterval(interval)
  }, [currentConversationId, addMessage])

  const sendMessage = useCallback(
    async (content: string) => {
      if (chatState === "streaming") return

      if (warmupPromiseRef.current) {
        await warmupPromiseRef.current
        warmupPromiseRef.current = null
      }

      setError(null)
      setCurrentToolCall(null)

      const userMessage: ChatMessage = {
        id: generateId(),
        role: "user",
        content,
        timestamp: new Date().toISOString(),
        toolResults: [],
        proposedAction: null,
        optionCards: [],
      }

      const assistantId = generateId()
      const assistantMessage: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: new Date().toISOString(),
        toolResults: [],
        proposedAction: null,
        optionCards: [],
      }

      let convId = currentConversationId ?? useChatStore.getState().currentConversationId
      let needsAutoTitle = false
      let isNewConversation = false
      if (!convId) {
        convId = `conv-${Date.now()}`
        addConversation({
          id: convId,
          title: "New conversation",
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
          messages: [],
        })
        isNewConversation = true
        needsAutoTitle = true
      } else {
        const conv = useChatStore.getState().conversations.find((c) => c.id === convId)
        const userMessageCount = conv?.messages.filter(m => m.role === "user").length || 0
        if (conv && conv.title === "New conversation" && userMessageCount === 0) {
          needsAutoTitle = true
        }
      }

      addMessage(convId, {
        id: userMessage.id,
        role: "user",
        content: userMessage.content,
        timestamp: userMessage.timestamp,
      })

      setMessages((prev) => [...prev, userMessage, assistantMessage])
      setChatState("streaming")

      try {
        const hadPriorSession = !!useChatStore.getState().getSessionId(convId)
        logger.info("sending to OpenCode", { conversationId: convId })
        const { chatModelSelection } = useSettingsStore.getState()
        const response = await sendOpenCodeMessage(convId, content, chatModelSelection)
        logger.info("OpenCode response received", {
          provider: response.info?.providerID,
          model: response.info?.modelID,
          parts: response.parts?.length,
        })

        const parsed = parseOpenCodeResponse(response)

        const sessionMessages = await fetchSessionMessages(convId)
        const session = extractSessionData(sessionMessages, parsed.toolResults)
        const toolResults = session.tools

        let phase: string | null = null
        const phaseTool = [...toolResults].reverse().find((t) => t.name === "signal_phase")
        if (phaseTool?.result) {
          try {
            const p = typeof phaseTool.result === "string" ? JSON.parse(phaseTool.result) : phaseTool.result
            if (p?.phase) phase = p.step ? `${p.phase}:${p.step}` : p.phase
          } catch { /* ignore */ }
        }
        const responseContent = session.text || parsed.content

        if (toolResults.length > 0) {
          setCurrentToolCall(toolResults[toolResults.length - 1]!)
        }

        let proposedAction: ProposedAction | null = null
        const validationTool = toolResults.find((t) => t.name in VALIDATE_TO_CREATE_ENDPOINT)
        if (validationTool) {
          const validated = extractValidatedJobConfig(validationTool.result)
          if (validated) {
            proposedAction = {
              action: `Create ${validated.jobType.toUpperCase()} Job`,
              description: `Submit this ${validated.jobType} job?`,
              params: validated.config,
              jobType: validated.jobType as "sdg" | "training",
              endpoint: VALIDATE_TO_CREATE_ENDPOINT[validationTool.name],
              config: validated.config,
              parentJobId: validated.parentJobId,
              recipe: validated.recipe,
            }
          }
        }

        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === assistantId)
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = {
            ...prev[idx]!,
            content: responseContent,
            toolResults,
            proposedAction,
            phase: phase ?? undefined,
          }
          return updated
        })

        addMessage(convId, {
          id: assistantId,
          role: "assistant",
          content: responseContent,
          timestamp: new Date().toISOString(),
          toolResults,
          proposedAction,
          phase: phase ?? undefined,
        })

        setChatState(proposedAction ? "action_pending" : "done")
        useChatStore.getState().setSessionStatus(convId, "connected")

        if (!hadPriorSession && messagesRef.current.length > 1) {
          useChatStore.getState().setSessionStatus(convId, "rebuilt")
        }

        if (needsAutoTitle && convId) {
          const titleConvId = convId
          generateChatTitle(content).then((title) => {
            updateConversationTitle(titleConvId, title)
          }).catch(() => {
            updateConversationTitle(titleConvId, content.slice(0, 40) + (content.length > 40 ? "..." : ""))
          })
        }
      } catch (err) {
        logger.error("OpenCode error", { error: err instanceof Error ? err.message : String(err) })
        setMessages((prev) => prev.filter((m) => m.id !== assistantId))
        setError(err instanceof Error ? err.message : "Unknown error")
        setChatState("error")
      }

      if (isNewConversation) {
        setCurrentConversationId(convId)
      }
    },
    [
      chatState,
      currentConversationId,
      addConversation,
      addMessage,
      updateConversationTitle,
      setCurrentConversationId,
    ],
  )

  const selectOption = useCallback((messageId: string, value: string) => {
    setMessages((prev) =>
      prev.map((m) =>
        m.id === messageId ? { ...m, selectedOptionValue: value } : m,
      ),
    )
    if (currentConversationId) {
      useChatStore.getState().updateMessageFields(currentConversationId, messageId, { selectedOptionValue: value })
    }
  }, [currentConversationId])

  const confirmAction = useCallback(async () => {
    const actionMsg = [...messagesRef.current].reverse().find((m) => m.proposedAction !== null)
    if (!actionMsg?.proposedAction) return

    const { endpoint, config, parentJobId, recipe, jobType } = actionMsg.proposedAction
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

      setMessages((prev) =>
        prev.map((m) =>
          m.id === actionMsg.id
            ? { ...m, proposedAction: null, toolResults: [...m.toolResults, jobToolResult] }
            : m,
        ),
      )

      if (currentConversationId) {
        useChatStore.getState().updateMessageFields(currentConversationId, actionMsg.id, {
          proposedAction: null,
          toolResults: [...actionMsg.toolResults, jobToolResult],
        })

        const sessionId = useChatStore.getState().getSessionId(currentConversationId)
        if (sessionId) {
          try {
            const resp = await fetch(`/agent/session/${sessionId}/message`, {
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
      }

      setChatState("done")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit job")
      setChatState("error")
    }
  }, [currentConversationId])

  const rejectAction = useCallback(async () => {
    const actionMsg = [...messagesRef.current].reverse().find((m) => m.proposedAction !== null)
    if (!actionMsg?.proposedAction) return

    setMessages((prev) =>
      prev.map((m) =>
        m.id === actionMsg.id ? { ...m, proposedAction: null } : m,
      ),
    )

    if (currentConversationId) {
      useChatStore.getState().updateMessageFields(currentConversationId, actionMsg.id, {
        proposedAction: null,
      })

      const sessionId = useChatStore.getState().getSessionId(currentConversationId)
      if (sessionId) {
        try {
          const resp = await fetch(`/agent/session/${sessionId}/message`, {
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
    }

    setChatState("done")
  }, [currentConversationId])

  const FOLLOW_UP_STATUSES = new Set(["running", "succeeded", "failed", "cancelled"])

  const handleJobStatusChange = useCallback(
    async (jobId: string, jobType: string, status: string) => {
      if (!FOLLOW_UP_STATUSES.has(status)) return

      const eventKey = `${jobId}:${status}`
      if (_firedJobEvents.has(eventKey)) return
      _firedJobEvents.add(eventKey)

      const convId = currentConversationId ?? useChatStore.getState().currentConversationId
      if (!convId) return

      const shortId = jobId.slice(0, 8)
      let prompt: string
      if (status === "running") {
        prompt = `[System event: The ${jobType} job ${shortId} is now running. You MUST call the present_options tool with 2-3 options (e.g. view job on Jobs page, set up training while waiting, continue chatting). Keep your text response to one sentence. Do NOT repeat the job details table.]`
      } else if (status === "succeeded") {
        prompt = `[System event: The ${jobType} job ${shortId} just succeeded. You MUST call the present_options tool with 2-3 next step options for the user (e.g. preview the generated data, start training, view job details). Keep your text response to one sentence. Do NOT repeat the job details table.]`
      } else {
        prompt = `[System event: The ${jobType} job ${shortId} just ${status}. You MUST call the present_options tool with 2-3 options (e.g. view job logs, retry, view job details). Keep your text response to one sentence. Do NOT repeat the job details table.]`
      }

      const assistantId = generateId()
      try {
        const { chatModelSelection } = useSettingsStore.getState()
        const response = await sendOpenCodeMessage(convId, prompt, chatModelSelection)
        const parsed = parseOpenCodeResponse(response)
        const sessionMessages = await fetchSessionMessages(convId)
        const session = extractSessionData(sessionMessages, parsed.toolResults)

        const followUp: ChatMessage = {
          id: assistantId,
          role: "assistant",
          content: session.text || parsed.content,
          timestamp: new Date().toISOString(),
          toolResults: session.tools,
          proposedAction: null,
          optionCards: [],
        }

        setMessages((prev) => [...prev, followUp])
        addMessage(convId, {
          id: assistantId,
          role: "assistant",
          content: followUp.content,
          timestamp: followUp.timestamp,
          toolResults: followUp.toolResults,
        })
      } catch {
        logger.warn("job follow-up failed", { jobId, status })
      }
    },
    [currentConversationId, addMessage],
  )

  const isStreaming = chatState === "streaming" || chatState === "tool_call"

  const latestAction = [...messages].reverse().find((m) => m.proposedAction !== null)?.proposedAction ?? null

  return {
    messages,
    sendMessage,
    selectOption,
    handleJobStatusChange,
    isStreaming,
    error,
    chatState,
    currentToolCall,
    proposedAction: latestAction,
    confirmAction,
    rejectAction,
  }
}
