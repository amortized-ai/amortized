import { useCallback, useEffect, useRef, useState } from "react"
import { sendOpenCodeMessage, fetchSessionMessages, generateChatTitle } from "@/lib/api-client"
import { useChatStore } from "@/stores/chat-store"
import { useSettingsStore } from "@/stores/settings-store"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-chat")
import type {
  ChatMessage,
  ChatState,
  ToolResult,
  OpenCodeResponse,
} from "../types"

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
  "estimate_sdg_cost",
  "compare_sdg_models",
  "estimate_training_cost",
  "estimate_training_method_cost",
  "estimate_eval_cost",
  "submit_recipe_job",
  "create_job",
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

  for (const msg of currentTurnMessages) {
    const info = (msg as unknown as Record<string, unknown>).info as Record<string, unknown> | undefined
    if (info?.role !== "assistant") continue

    for (const part of msg.parts) {
      if (part.type === "text" && part.text) {
        textParts.push(part.text)
      } else if (part.type === "tool") {
        const name = normalizeToolName(part.tool ?? "")
        if (UI_TOOLS.has(name) && !seen.has(name.toLowerCase())) {
          seen.add(name.toLowerCase())
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

  const sendMessage = useCallback(
    async (content: string) => {
      if (chatState === "streaming") return

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

        const phaseTool = toolResults.find((t) => t.name === "signal_phase")
        let phase: string | null = null
        if (phaseTool?.result) {
          try {
            const p = typeof phaseTool.result === "string" ? JSON.parse(phaseTool.result) : phaseTool.result
            if (p?.phase) phase = p.step ? `${p.phase}:${p.step}` : p.phase
          } catch { /* ignore */ }
        }
        const responseContent = parsed.content || session.text

        if (toolResults.length > 0) {
          setCurrentToolCall(toolResults[toolResults.length - 1]!)
        }

        setMessages((prev) => {
          const idx = prev.findIndex((m) => m.id === assistantId)
          if (idx === -1) return prev
          const updated = [...prev]
          updated[idx] = {
            ...prev[idx]!,
            content: responseContent,
            toolResults,
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
          phase: phase ?? undefined,
        })

        setChatState("done")
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

  const confirmAction = useCallback(async () => {}, [])
  const rejectAction = useCallback(async () => {}, [])

  const isStreaming = chatState === "streaming" || chatState === "tool_call"

  return {
    messages,
    sendMessage,
    selectOption,
    isStreaming,
    error,
    chatState,
    currentToolCall,
    proposedAction: null,
    confirmAction,
    rejectAction,
  }
}
