import { Card, CardContent } from "@/components/ui/card"
import { Zap, Star, Brain, Sparkles } from "lucide-react"
import { cn } from "@/lib/utils"
import { friendlyModelName } from "../utils/workflow-options"

interface ModelOption {
  model_id: string
  label: string
  description: string
  total_cost: number
  per_sample_cost: number
}

interface ModelComparisonEstimate {
  num_samples: number
  models: ModelOption[]
}

function formatUSD(value: number): string {
  const n = Number(value)
  if (n < 0.01 && n > 0) return "< $0.01"
  return `$${n.toFixed(2)}`
}

function getModelIcon(label: string): typeof Zap {
  const lower = label.toLowerCase()
  if (lower.includes("haiku")) return Zap
  if (lower.includes("sonnet")) return Star
  if (lower.includes("gpt")) return Brain
  return Sparkles
}

function getModelColor(label: string): string {
  const lower = label.toLowerCase()
  if (lower.includes("haiku")) return "text-rh-orange"
  if (lower.includes("sonnet")) return "text-rh-purple"
  if (lower.includes("gpt") || lower.includes("openai")) return "text-rh-blue"
  return "text-rh-teal"
}

interface ModelComparisonCardProps {
  estimate: ModelComparisonEstimate
}

export function ModelComparisonCard({ estimate }: ModelComparisonCardProps) {
  const cheapest = estimate.models.length > 0
    ? estimate.models.reduce((min, m) => m.total_cost < min.total_cost ? m : min, estimate.models[0]!)
    : null

  return (
    <Card className="border-rh-purple/30 dark:border-rh-purple-dark/50 bg-rh-purple-light/30 dark:bg-rh-purple-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm font-semibold text-foreground">
            Model Cost Comparison
          </span>
          <span className="text-xs text-muted-foreground">
            {estimate.num_samples.toLocaleString()} samples
          </span>
        </div>

        <div className="space-y-2">
          {estimate.models.map((model) => {
            const friendlyLabel = friendlyModelName(model.label)
            const Icon = getModelIcon(friendlyLabel)
            const color = getModelColor(friendlyLabel)
            const isCheapest = cheapest !== null && model.model_id === cheapest.model_id

            return (
              <div
                key={model.model_id}
                className={cn(
                  "flex items-center justify-between rounded-lg border px-3 py-2.5",
                  isCheapest
                    ? "border-rh-green/30 dark:border-rh-green-dark/50 bg-rh-green-light/50 dark:bg-rh-green-dark/20"
                    : "border-rh-purple-light dark:border-rh-purple-dark/30 bg-background/60 dark:bg-rh-gray-90/30",
                )}
              >
                <div className="flex items-center gap-2.5 flex-1 min-w-0">
                  <Icon className={cn("h-4 w-4 shrink-0", color)} />
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-foreground">{friendlyLabel}</span>
                      {isCheapest && (
                        <span className="rounded-full bg-rh-green-light dark:bg-rh-green-dark/40 px-1.5 py-0.5 text-[10px] font-semibold text-rh-green-dark dark:text-rh-green">
                          Best value
                        </span>
                      )}
                    </div>
                    {model.description && (
                      <p className="text-xs text-muted-foreground">{model.description}</p>
                    )}
                  </div>
                </div>
                <div className="text-right ml-3 shrink-0">
                  <p className="text-sm font-bold tabular-nums text-foreground">
                    {formatUSD(model.total_cost)}
                  </p>
                  <p className="text-[10px] tabular-nums text-muted-foreground">
                    {formatUSD(model.per_sample_cost)}/sample
                  </p>
                </div>
              </div>
            )
          })}
        </div>

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Prices from OpenRouter &middot; estimates based on ~500 input + 300 output tokens/sample
        </p>
      </CardContent>
    </Card>
  )
}
