import { useMemo, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import { Bot } from "lucide-react"
import type { ChatMessage, OptionCard } from "../types"
import { ToolActivity } from "./tool-badge"
import { ActionCard } from "./action-card"
import { OptionCards } from "./option-cards"
import { CostAnalysisCard } from "./cost-analysis-card"
import { TrainingCostCard } from "./training-cost-card"
import { ModelComparisonCard } from "./model-comparison-card"
import { TrainingMethodCostCard } from "./training-method-cost-card"
import { JobMonitorCard } from "./job-monitor-card"

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

  // Extract job ID and type from job submission tool result
  const { jobId, jobType } = useMemo(() => {
    if (isUser) return { jobId: null, jobType: "SDG" }
    const submitTool = message.toolResults.find(t => t.name === "submit_recipe_job" || t.name === "create_job")
    if (!submitTool?.result) return { jobId: null, jobType: "SDG" }
    try {
      const parsed = typeof submitTool.result === "string"
        ? JSON.parse(submitTool.result)
        : submitTool.result
      const id = parsed?.id ?? null
      const type = parsed?.type ? String(parsed.type).toUpperCase() : "SDG"
      return { jobId: id, jobType: type }
    } catch {
      return { jobId: null, jobType: "SDG" }
    }
  }, [isUser, message.toolResults])

  const [monitorDismissed, setMonitorDismissed] = useState(false)

  const costEstimate = useMemo(() => {
    if (isUser) return null
    const costTool = message.toolResults.find(t => t.name === "estimate_sdg_cost")
    if (!costTool?.result) return null
    try {
      const parsed = typeof costTool.result === "string"
        ? JSON.parse(costTool.result)
        : costTool.result
      if (parsed?.cost && parsed?.comparison) return parsed
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const trainingCostEstimate = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find(t => t.name === "estimate_training_cost")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string"
        ? JSON.parse(tool.result)
        : tool.result
      if (parsed?.models && Array.isArray(parsed.models)) return parsed
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const trainingMethodCost = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find(t => t.name === "estimate_training_method_cost")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string"
        ? JSON.parse(tool.result)
        : tool.result
      if (parsed?.methods && Array.isArray(parsed.methods)) return parsed
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const modelComparison = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find(t => t.name === "compare_sdg_models")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string"
        ? JSON.parse(tool.result)
        : tool.result
      if (parsed?.models && Array.isArray(parsed.models)) return parsed
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const visibleToolResults = useMemo(() => {
    const hidden = new Set<string>(["signal_phase"])
    if (costEstimate) hidden.add("estimate_sdg_cost")
    if (trainingCostEstimate) hidden.add("estimate_training_cost")
    if (trainingMethodCost) hidden.add("estimate_training_method_cost")
    if (modelComparison) hidden.add("compare_sdg_models")
    if (structuredOptions) hidden.add("present_options")
    return message.toolResults.filter((t) => !hidden.has(t.name))
  }, [message.toolResults, costEstimate, trainingCostEstimate, trainingMethodCost, modelComparison, structuredOptions])


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

        {costEstimate && (
          <div className="mt-3">
            <CostAnalysisCard estimate={costEstimate} />
          </div>
        )}

        {trainingCostEstimate && (
          <div className="mt-3">
            <TrainingCostCard estimate={trainingCostEstimate} />
          </div>
        )}

        {trainingMethodCost && (
          <div className="mt-3">
            <TrainingMethodCostCard estimate={trainingMethodCost} />
          </div>
        )}

        {modelComparison && (
          <div className="mt-3">
            <ModelComparisonCard estimate={modelComparison} />
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
