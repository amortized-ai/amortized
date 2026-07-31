import { useMemo, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import { Bot } from "lucide-react"
import type { ChatMessage, OptionCard } from "../types"
import { ToolActivity } from "./tool-badge"
import { ActionCard } from "./action-card"
import { OptionCards } from "./option-cards"
import { ModelPricingCard } from "./model-pricing-card"
import { VRAMEstimateCard } from "./vram-estimate-card"
import { JobMonitorCard } from "./job-monitor-card"
import { extractJobInfo } from "../utils/parse-tool-result"

const TOOL_XML_RE =
  /<(?:function_calls|function_response|antml:function_calls|antml:invoke)[^>]*>[\s\S]*?<\/(?:function_calls|function_response|antml:function_calls|antml:invoke)>/g

function stripToolXml(text: string): string {
  return text.replace(TOOL_XML_RE, "").replace(/\n{3,}/g, "\n\n").trim()
}

interface MessageBubbleProps {
  message: ChatMessage
  isLatest?: boolean
  onOptionSelect?: (value: string) => void
  onConfirmAction?: () => void
  onRejectAction?: () => void
}

export function MessageBubble({
  message,
  isLatest,
  onOptionSelect,
  onConfirmAction,
  onRejectAction,
}: MessageBubbleProps) {
  const isUser = message.role === "user"

  const displayContent = useMemo(
    () => (isUser ? message.content : stripToolXml(message.content)),
    [isUser, message.content],
  )

  const structuredOptions = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find(t => t.name === "present_options")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string" ? JSON.parse(tool.result) : tool.result
      if (parsed?.options && Array.isArray(parsed.options)) return parsed.options as OptionCard[]
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const parsedOptions = useMemo(() => {
    if (isUser || message.optionCards.length > 0) return []
    return structuredOptions ?? []
  }, [isUser, message.optionCards.length, structuredOptions])

  const { jobId, jobType } = useMemo(() => {
    if (isUser) return { jobId: null, jobType: "SDG" }
    const submitTools = message.toolResults.filter(t => t.name === "submit_recipe_job" || t.name === "create_job")
    for (const tool of submitTools) {
      if (!tool.result) continue
      const info = extractJobInfo(tool.result)
      if (info.jobId) return info
    }
    return { jobId: null, jobType: "SDG" }
  }, [isUser, message.toolResults])

  const [monitorDismissed, setMonitorDismissed] = useState(false)

  const modelPricing = useMemo(() => {
    if (isUser) return null
    const tools = message.toolResults.filter(
      t => t.name === "get_model_pricing" || t.name === "show_model_pricing"
    )
    if (tools.length === 0) return null
    const allModels: Record<string, unknown>[] = []
    const seen = new Set<string>()
    for (const tool of tools) {
      if (!tool.result) continue
      try {
        const parsed = typeof tool.result === "string" ? JSON.parse(tool.result) : tool.result
        for (const m of parsed?.models ?? []) {
          if (m.model_id && !seen.has(m.model_id)) {
            seen.add(m.model_id)
            allModels.push(m)
          }
        }
      } catch { /* ignore */ }
    }
    if (allModels.length === 0) return null
    return { models: allModels }
  }, [isUser, message.toolResults])

  const vramEstimate = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find(t => t.name === "show_vram_estimate")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string"
        ? JSON.parse(tool.result)
        : tool.result
      if (parsed?.estimates && Array.isArray(parsed.estimates)) return parsed
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const visibleToolResults = useMemo(() => {
    const hidden = new Set<string>(["signal_phase", "get_document_sections", "get_section_content", "get_document_content"])
    if (modelPricing) {
      hidden.add("get_model_pricing")
      hidden.add("show_model_pricing")
    }
    if (vramEstimate) {
      hidden.add("estimate_training_resources")
      hidden.add("show_vram_estimate")
    }
    if (structuredOptions) hidden.add("present_options")
    return message.toolResults.filter((t) => !hidden.has(t.name))
  }, [message.toolResults, modelPricing, vramEstimate, structuredOptions])


  return (
    <div
      className={cn(
        "group/msg flex w-full gap-3",
        isLatest && "animate-message-in",
        isUser ? "justify-end" : "justify-start",
      )}
    >
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-rh-red mt-0.5 shadow-sm">
          <Bot className="h-4 w-4 text-white" />
        </div>
      )}
      <div
        className={cn(
          "max-w-2xl rounded-2xl px-4 py-3 transition-colors duration-200",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-md hover:bg-primary/90"
            : "bg-muted/60 text-foreground border border-border/50 rounded-bl-md hover:bg-muted/80",
        )}
      >
        {!isUser && !displayContent && (
          <div className="flex items-center gap-1.5 py-1 px-1">
            <span className="thinking-dot h-2 w-2 rounded-full bg-muted-foreground/60" style={{ animationDelay: "0ms" }} />
            <span className="thinking-dot h-2 w-2 rounded-full bg-muted-foreground/60" style={{ animationDelay: "280ms" }} />
            <span className="thinking-dot h-2 w-2 rounded-full bg-muted-foreground/60" style={{ animationDelay: "560ms" }} />
          </div>
        )}
        {displayContent && (
          isUser ? (
            <div className="whitespace-pre-wrap text-sm">{displayContent}</div>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none [&_table]:text-xs [&_th]:px-3 [&_th]:py-1.5 [&_td]:px-3 [&_td]:py-1.5 [&_table]:rounded-lg [&_table]:overflow-hidden [&_thead]:bg-muted/50 [&_strong]:text-foreground">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {displayContent}
              </ReactMarkdown>
            </div>
          )
        )}

        {modelPricing && (
          <div className="mt-3">
            <ModelPricingCard data={modelPricing} />
          </div>
        )}

        {vramEstimate && (
          <div className="mt-3">
            <VRAMEstimateCard data={vramEstimate} />
          </div>
        )}
        {visibleToolResults.length > 0 && (
          <ToolActivity toolResults={visibleToolResults} />
        )}

        {message.proposedAction && onConfirmAction && onRejectAction && (
          <div className="mt-3">
            <ActionCard
              action={message.proposedAction}
              onConfirm={onConfirmAction}
              onReject={onRejectAction}
            />
          </div>
        )}

        {message.optionCards.length > 0 && onOptionSelect && (
          <div className="mt-3">
            <OptionCards
              cards={message.optionCards}
              onSelect={onOptionSelect}
              selectedValue={message.selectedOptionValue}
            />
          </div>
        )}

        {/* Job monitoring card - only for latest message with submit_recipe_job */}
        {!monitorDismissed && jobId && (
          <div className="mt-3">
            <JobMonitorCard
              jobId={jobId}
              jobType={jobType}
              onDismiss={() => setMonitorDismissed(true)}
            />
          </div>
        )}

        {parsedOptions.length > 0 && onOptionSelect && (
          <div className="mt-3">
            <OptionCards cards={parsedOptions} onSelect={onOptionSelect} selectedValue={message.selectedOptionValue} />
          </div>
        )}
      </div>
    </div>
  )
}
