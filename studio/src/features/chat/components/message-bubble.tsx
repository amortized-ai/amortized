import { useMemo, useState } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import { Bot } from "lucide-react"
import type { ChatMessage, OptionCard } from "../types"
import { ToolActivity } from "./tool-badge"
import { ActionCard } from "./action-card"
import { OptionCards } from "./option-cards"
import { detectWorkflowStep } from "../utils/workflow-options"
import { useGatewayRoutes } from "@/features/settings"
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

const NUMBERED_OPTION_RE = /^\d+\)\s+(.+)/
const BULLET_OPTION_RE = /^[-•]\s+(.+)/
const DASH_OPTION_RE = /^(?:\*\*)?(.+?)(?:\*\*)?\s*—\s+(.+)/

/**
 * Strip markdown inline formatting so length checks reflect visible text.
 */
function stripMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/`(.+?)`/g, "$1")
}

function isActionSuggestion(text: string): boolean {
  const plain = stripMarkdown(text.trim())
  if (plain.length > 140) return false
  if (/^[^—:]+:\s/.test(plain)) return false
  return true
}

function parseOptionsFromText(content: string): OptionCard[] {
  const lines = content.split("\n").map((l) => l.trim()).filter(Boolean)
  let currentGroup: OptionCard[] = []
  let lastGroup: OptionCard[] = []

  for (const line of lines) {
    const numbered = NUMBERED_OPTION_RE.exec(line)
    const bullet = !numbered ? BULLET_OPTION_RE.exec(line) : null
    const match = numbered ?? bullet
    if (match?.[1]) {
      const text = match[1]
      const dashParts = text.split(/\s*—\s*/)
      const title = stripMarkdown(dashParts[0]!.trim())
      const description = dashParts.length > 1 ? dashParts.slice(1).join(" — ").trim() : ""
      if (title.length > 120) continue
      currentGroup.push({ title, description, value: text })
    } else {
      const dashMatch = DASH_OPTION_RE.exec(line)
      if (dashMatch?.[1] && dashMatch[2]) {
        const title = stripMarkdown(dashMatch[1].trim())
        const description = dashMatch[2].trim()
        if (title.length > 60) continue
        if (description.length > 80) continue
        if (/^(perfect|great|awesome|excellent|sure|ok|got it|nice|wonderful)/i.test(title)) continue
        currentGroup.push({ title, description, value: `${title} — ${description}` })
      } else if (currentGroup.length > 0) {
        if (currentGroup.length >= 2) lastGroup = currentGroup
        currentGroup = []
      }
    }
  }
  if (currentGroup.length >= 2) lastGroup = currentGroup

  return lastGroup
}

interface MessageBubbleProps {
  message: ChatMessage
  isLatest?: boolean
  allMessages?: ChatMessage[]
  onOptionSelect?: (value: string) => void
  onConfirmAction?: () => void
  onRejectAction?: () => void
}

export function MessageBubble({
  message,
  isLatest,
  allMessages,
  onOptionSelect,
  onConfirmAction,
  onRejectAction,
}: MessageBubbleProps) {
  const isUser = message.role === "user"

  const displayContent = useMemo(
    () => (isUser ? message.content : stripToolXml(message.content)),
    [isUser, message.content],
  )

  const { data: gatewayRoutes } = useGatewayRoutes()

  const parsedOptions = useMemo(() => {
    if (isUser || message.optionCards.length > 0) return []
    const workflow = detectWorkflowStep(displayContent, allMessages, gatewayRoutes)
    if (workflow) return workflow.cards
    return parseOptionsFromText(displayContent)
  }, [isUser, displayContent, message.optionCards.length, allMessages, gatewayRoutes])

  // Filter explicit optionCards: only action-oriented items become cards;
  // the rest are rendered as plain text lines below the message.
  const filteredOptionCards = useMemo(
    () => message.optionCards.filter((c) => isActionSuggestion(c.title)),
    [message.optionCards],
  )
  const nonActionCards = useMemo(
    () => message.optionCards.filter((c) => !isActionSuggestion(c.title)),
    [message.optionCards],
  )

  // Extract job ID, type, and MLflow run ID from submit_recipe_job tool results
  const { jobId, jobType, mlflowRunId } = useMemo(() => {
    if (isUser) return { jobId: null, jobType: "SDG", mlflowRunId: null }
    let id: string | null = null
    let type = "SDG"
    let runId: string | null = null

    const submitTool = message.toolResults.find(t =>
      t.name === "submit_recipe_job" || t.name === "submit recipe job"
    )
    if (submitTool?.result) {
      try {
        const parsed = typeof submitTool.result === "string"
          ? JSON.parse(submitTool.result)
          : submitTool.result
        id = parsed?.id ?? null
        if (parsed?.type) type = String(parsed.type).toUpperCase()
        if (parsed?.mlflow_run_id) runId = String(parsed.mlflow_run_id)
      } catch { /* fall through to text extraction */ }
    }

    if (!id) {
      const plain = stripMarkdown(displayContent)
      const idMatch = /Job ID[:\s|]+([0-9a-f-]{36})/i.exec(plain)
      if (idMatch) id = idMatch[1]!
    }

    // Infer job type ONLY from explicit "your X job" or "X job submitted" phrasing
    if (type === "SDG") {
      const lower = displayContent.toLowerCase()
      if (
        /your training job/i.test(lower) ||
        /training job (is |submitted|queued|running)/i.test(lower)
      ) {
        type = "TRAINING"
      }
    }

    return { jobId: id, jobType: type, mlflowRunId: runId }
  }, [isUser, message.toolResults, displayContent])

  const [monitorDismissed, setMonitorDismissed] = useState(false)
  const [completedStatus, setCompletedStatus] = useState<string | null>(null)

  const jobOptionCards = useMemo(() => {
    if (!jobId) return []
    const viewJobCard = { title: "View Job", description: "Open the job in the Jobs page", value: `__nav:/jobs?job=${encodeURIComponent(jobId)}` } as OptionCard

    if (completedStatus === "failed" || completedStatus === "cancelled") {
      const cards: OptionCard[] = [
        viewJobCard,
        { title: "View Logs", description: "Check what went wrong", value: `__nav:/jobs?job=${encodeURIComponent(jobId)}&tab=logs` },
        { title: "Try again", description: "Resubmit with the same settings", value: "Let's try running that job again with the same settings" },
      ]
      return cards
    }

    if (completedStatus === "succeeded") {
      if (jobType === "TRAINING") {
        const modelNav = mlflowRunId ? `__nav:/models?run=${encodeURIComponent(mlflowRunId)}` : "__nav:/models"
        return [
          { title: "View Model", description: "Browse trained model artifacts", value: modelNav },
          { title: "Train with different settings", description: "Adjust model, method, or hyperparameters", value: "I'd like to train again with different settings" },
          { title: "I'm done", description: "That's all I needed", value: "I'm done for now, thanks!" },
        ] as OptionCard[]
      }
      return [
        { title: "Generate more samples", description: "Create a larger dataset with more diverse coverage", value: "Generate more samples with broader coverage" },
        { title: "Continue to training", description: "Fine-tune a student model on this data", value: "Let's continue to the training step" },
        { title: "I'm done", description: "That's all I needed", value: "I'm done for now, thanks!" },
      ] as OptionCard[]
    }

    return [viewJobCard]
  }, [jobId, jobType, completedStatus, mlflowRunId])

  const costEstimate = useMemo(() => {
    if (isUser) return null
    const costTool = message.toolResults.find(
      (t) => t.name === "estimate_sdg_cost" || t.name === "estimate sdg cost"
    )
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
    const tool = message.toolResults.find(
      (t) => t.name === "estimate_training_cost" || t.name === "estimate training cost"
    )
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
    const tool = message.toolResults.find(
      (t) => t.name === "estimate_training_method_cost" || t.name === "estimate training method cost"
    )
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
    const tool = message.toolResults.find(
      (t) => t.name === "compare_sdg_models" || t.name === "compare sdg models"
    )
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string"
        ? JSON.parse(tool.result)
        : tool.result
      if (parsed?.models && Array.isArray(parsed.models)) return parsed
    } catch { /* ignore parse errors */ }
    return null
  }, [isUser, message.toolResults])

  const trainingCostSummary = useMemo(() => {
    if (isUser) return null
    const tool = message.toolResults.find((t) => t.name === "training cost summary")
    if (!tool?.result) return null
    try {
      const parsed = typeof tool.result === "string" ? JSON.parse(tool.result) : tool.result
      if (parsed?.comparison) return parsed
    } catch { /* ignore */ }
    return null
  }, [isUser, message.toolResults])

  const visibleToolResults = useMemo(() => {
    const hidden = new Set<string>()
    hidden.add("signal_progress")
    hidden.add("signal progress")
    hidden.add("signal_phase")
    hidden.add("signal phase")
    hidden.add("present_options")
    hidden.add("present options")
    if (trainingCostSummary) hidden.add("training cost summary")
    if (costEstimate) {
      hidden.add("estimate_sdg_cost")
      hidden.add("estimate sdg cost")
    }
    if (trainingCostEstimate) {
      hidden.add("estimate_training_cost")
      hidden.add("estimate training cost")
    }
    if (trainingMethodCost) {
      hidden.add("estimate_training_method_cost")
      hidden.add("estimate training method cost")
    }
    if (modelComparison) {
      hidden.add("compare_sdg_models")
      hidden.add("compare sdg models")
    }
    if (hidden.size === 0) return message.toolResults
    return message.toolResults.filter((t) => !hidden.has(t.name))
  }, [message.toolResults, costEstimate, trainingCostEstimate, trainingMethodCost, modelComparison])


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
        {trainingCostSummary?.comparison && (
          <div className="mt-3">
            <CostAnalysisCard
              phase="training"
              estimate={{
                model_label: trainingCostSummary.model_label || "LoRA SFT",
                num_samples: trainingCostSummary.num_samples || 100,
                cost: { input: 0, output: 0, total: trainingCostSummary.methods?.[0]?.estimated_cost ?? 0 },
                comparison: {
                  manual_labeling_total: trainingCostSummary.comparison.manual_training_total,
                  savings_amount: trainingCostSummary.comparison.savings_amount,
                  savings_percent: trainingCostSummary.comparison.savings_percent,
                },
              }}
            />
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

        {nonActionCards.length > 0 && (
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
            {nonActionCards.map((card) => (
              <li key={card.value}>{card.title}</li>
            ))}
          </ul>
        )}

        {filteredOptionCards.length > 0 && onOptionSelect && (
          <div className="mt-3">
            <OptionCards
              cards={filteredOptionCards}
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
              onComplete={(s) => setCompletedStatus(s)}
            />
          </div>
        )}

        {jobOptionCards.length > 0 && onOptionSelect ? (
          <div className="mt-3">
            <OptionCards cards={jobOptionCards} onSelect={onOptionSelect} selectedValue={message.selectedOptionValue} />
          </div>
        ) : parsedOptions.length > 0 && onOptionSelect ? (
          <div className="mt-3">
            <OptionCards cards={parsedOptions} onSelect={onOptionSelect} selectedValue={message.selectedOptionValue} />
          </div>
        ) : null}
      </div>
    </div>
  )
}
