import { useCallback, useEffect, useRef, useState } from "react"
import { sendOpenCodeMessage, generateChatTitle } from "@/lib/api-client"
import { useChatStore } from "@/stores/chat-store"
import { useSettingsStore } from "@/stores/settings-store"
import { useGatewayRoutes } from "@/features/settings"
import { getLogger } from "@/lib/logger"

const logger = getLogger("use-chat")
import type {
  ChatMessage,
  ChatState,
  ToolResult,
  OpenCodeResponse,
} from "../types"
import { autoCostEstimate } from "../utils/auto-cost"
import { extractPhase } from "../utils/workflow-options"

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
    seen.add(m[1]!.replace(/^mcp_amortized__/, "").replaceAll("_", " "))
  }
  const tools: ToolResult[] = [...seen].map((name) => ({
    name,
    result: "",
    collapsed: true,
  }))
  const cleaned = text
    .replace(TOOL_BLOCK_RE, "")
    .replace(STRAY_TAG_RE, "")
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
        name: part.tool.replace(/^mcp_amortized__/, "").replaceAll("_", " "),
        result: String(part.output ?? part.state ?? ""),
        collapsed: true,
      })
    }
  }

  const { cleanText, tools } = extractToolCalls(rawText)
  const seen = new Set(toolResults.map((t) => t.name.replace(/_/g, " ").toLowerCase()))
  for (const t of tools) {
    const key = t.name.replace(/_/g, " ").toLowerCase()
    if (!seen.has(key)) {
      seen.add(key)
      toolResults.push(t)
    }
  }

  return { content: cleanText, toolResults }
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

  const { data: gatewayRoutes } = useGatewayRoutes()
  const gatewayRoutesRef = useRef(gatewayRoutes)
  useEffect(() => { gatewayRoutesRef.current = gatewayRoutes }, [gatewayRoutes])

  const [currentToolCall, setCurrentToolCall] = useState<ToolResult | null>(null)

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
        const { cleanText: responseContent, phase } = extractPhase(parsed.content)
        const toolResults = parsed.toolResults

        try {
          const allMessages = [...messagesRef.current, userMessage]
          const autoCost = await autoCostEstimate(allMessages, responseContent, phase, gatewayRoutesRef.current)
          if (autoCost) {
            toolResults.push(autoCost)
          }
        } catch {
          /* auto-cost is best-effort */
        }

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
