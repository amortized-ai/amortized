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
import { ThinkingSteps } from "./thinking-steps"
import { extractJobInfo } from "../utils/parse-tool-result"

const TOOL_XML_RE =
  /<(?:function_calls|function_response|antml:function_calls|antml:invoke)[^>]*>[\s\S]*?<\/(?:function_calls|function_response|antml:function_calls|antml:invoke)>/g
const JOB_TOOL_NAMES = new Set(["submit_recipe_job", "create_sdg_job", "create_training_job", "create_job"])

function stripToolXml(text: string): string {
  return text.replace(TOOL_XML_RE, "").replace(/\n{3,}/g, "\n\n").trim()
}

interface MessageBubbleProps {
  message: ChatMessage
  isLatest?: boolean
  thinkingStep?: string | null
  onOptionSelect?: (value: string, messageId?: string) => void
  onConfirmAction?: () => void
  onRejectAction?: () => void
  onJobComplete?: (jobId: string, jobType: string, status: string) => void
}

export function MessageBubble({
  message,
  isLatest,
  thinkingStep,
  onOptionSelect,
  onConfirmAction,
  onRejectAction,
  onJobComplete,
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
      if (parsed?.options && Array.isArray(parsed.options)) {
        const options = (parsed.options as OptionCard[]).map((opt) => ({
          ...opt,
          value: opt.value ?? (opt.description ? `${opt.title} — ${opt.description}` : opt.title),
        }))
        const question =
          typeof parsed.question === "string" && parsed.question.trim()
            ? (parsed.question as string)
            : null
        return { question, options }
      }
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const jobSubmissions = useMemo(() => {
    if (isUser) return []
    return message.toolResults
      .filter((t) => JOB_TOOL_NAMES.has(t.name))
      .map((t) => {
        const info = extractJobInfo(t.result)
        return info.jobId ? { id: info.jobId, type: info.jobType } : null
      })
      .filter((j): j is { id: string; type: string } => !!j)
  }, [isUser, message.toolResults])

  const [dismissedJobs, setDismissedJobs] = useState<Set<string>>(new Set())

  const parsedOptions = useMemo(() => {
    if (isUser || message.optionCards.length > 0) return []
    return structuredOptions?.options ?? []
  }, [isUser, message.optionCards.length, structuredOptions])

  const modelPricing = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find(t => t.name === "show_model_pricing")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string" ? JSON.parse(tool.result) : tool.result
      if (parsed?.models && Array.isArray(parsed.models)) return parsed
    } catch { /* ignore */ }
    return null
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
    const hidden = new Set<string>([
      "signal_phase",
      "get_document_sections",
      "get_section_content",
      "get_document_content",
    ])
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

  // Some models (e.g. gpt-5.x) put their prose in the present_options `question` field and
  // return an empty final-answer text, so surface the question when there's no text content.
  const optionsQuestion = structuredOptions?.question ?? null

  // Only show the "thinking" dots when there is genuinely nothing to render yet. A completed
  // turn that produced only tool output (options/question/tools) must render that, not dots.
  const hasRenderableBody =
    !!displayContent ||
    !!optionsQuestion ||
    parsedOptions.length > 0 ||
    message.optionCards.length > 0 ||
    jobSubmissions.length > 0 ||
    visibleToolResults.length > 0 ||
    !!modelPricing ||
    !!vramEstimate ||
    !!message.proposedAction

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
          "rounded-2xl px-4 py-3 transition-colors duration-200",
          (parsedOptions.length > 0 || message.optionCards.length > 0 || jobSubmissions.length > 0) ? "max-w-3xl" : "max-w-2xl",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-md hover:bg-primary/90"
            : "bg-muted/60 text-foreground border border-border/50 rounded-bl-md hover:bg-muted/80",
        )}
      >
        {!isUser && !hasRenderableBody && (
          <div>
            <div className="flex items-center gap-1.5 py-1 px-1">
              <span className="thinking-dot h-2 w-2 rounded-full bg-muted-foreground/60" style={{ animationDelay: "0ms" }} />
              <span className="thinking-dot h-2 w-2 rounded-full bg-muted-foreground/60" style={{ animationDelay: "280ms" }} />
              <span className="thinking-dot h-2 w-2 rounded-full bg-muted-foreground/60" style={{ animationDelay: "560ms" }} />
            </div>
            <ThinkingSteps step={thinkingStep ?? null} />
          </div>
        )}
        {!isUser && !displayContent && optionsQuestion && (
          <div className="prose prose-sm dark:prose-invert max-w-none [&_strong]:text-foreground">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{optionsQuestion}</ReactMarkdown>
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
              onSelect={(v) => onOptionSelect(v, message.id)}
              selectedValue={message.selectedOptionValue}
            />
          </div>
        )}

        {jobSubmissions
          .filter((job) => !dismissedJobs.has(job.id))
          .map((job) => (
            <div key={job.id} className="mt-3">
              <JobMonitorCard
                jobId={job.id}
                jobType={job.type}
                onDismiss={() => setDismissedJobs((s) => new Set([...s, job.id]))}
                onComplete={onJobComplete}
              />
            </div>
          ))}

        {parsedOptions.length > 0 && onOptionSelect && (
          <div className="mt-3">
            <OptionCards cards={parsedOptions} onSelect={(v) => onOptionSelect(v, message.id)} selectedValue={message.selectedOptionValue} />
          </div>
        )}
      </div>
    </div>
  )
}
