import { Card, CardContent } from "@/components/ui/card"
import { Scale, CircleCheck } from "lucide-react"
import { cn } from "@/lib/utils"
import { friendlyModelName } from "../utils/workflow-options"

interface JudgeModelOption {
  model_id: string
  label: string
  description?: string
  total_cost: number
  per_sample_cost?: number
  recommended: boolean
}

interface EvalCostEstimate {
  judge_model: string
  judge_model_label: string
  num_samples: number
  input_tokens: number
  output_tokens: number
  input_cost: number
  output_cost: number
  total_cost: number
  cost_per_sample: number
  comparison: JudgeModelOption[]
}

function formatUSD(value: number): string {
  const n = Number(value)
  if (n < 0.01 && n > 0) return "< $0.01"
  return `$${n.toFixed(2)}`
}

interface EvalCostCardProps {
  estimate: EvalCostEstimate
}

export function EvalCostCard({ estimate }: EvalCostCardProps) {
  return (
    <Card className="border-rh-teal/30 dark:border-rh-teal-dark/50 bg-rh-teal-light/30 dark:bg-rh-teal-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <Scale className="h-4 w-4 text-rh-teal dark:text-rh-teal" />
          <span className="text-sm font-semibold text-foreground">Evaluation Cost Estimate</span>
          <span className="ml-auto text-xs text-muted-foreground">
            {friendlyModelName(estimate.judge_model_label)} &middot; {estimate.num_samples.toLocaleString()} samples
          </span>
        </div>

        <div className="grid grid-cols-3 gap-3 text-xs mb-3">
          <div className="space-y-1">
            <p className="font-medium text-muted-foreground">Total Cost</p>
            <p className="text-lg font-bold tabular-nums text-foreground">{formatUSD(estimate.total_cost)}</p>
          </div>
          <div className="space-y-1">
            <p className="font-medium text-muted-foreground">Input Tokens</p>
            <p className="text-sm font-semibold tabular-nums text-foreground">{estimate.input_tokens.toLocaleString()}</p>
            <p className="text-muted-foreground/70">{formatUSD(estimate.input_cost)}</p>
          </div>
          <div className="space-y-1">
            <p className="font-medium text-muted-foreground">Output Tokens</p>
            <p className="text-sm font-semibold tabular-nums text-foreground">{estimate.output_tokens.toLocaleString()}</p>
            <p className="text-muted-foreground/70">{formatUSD(estimate.output_cost)}</p>
          </div>
        </div>

        {estimate.comparison.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Judge Model Options</p>
            {estimate.comparison.map((model) => (
              <div
                key={model.model_id}
                className={cn(
                  "flex items-center justify-between rounded-lg border px-3 py-2.5",
                  model.recommended
                    ? "border-rh-green/30 dark:border-rh-green-dark/50 bg-rh-green-light/50 dark:bg-rh-green-dark/20"
                    : "border-rh-teal-light dark:border-rh-teal-dark/30 bg-background/60 dark:bg-rh-gray-90/30",
                )}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-foreground">{friendlyModelName(model.label)}</span>
                    {model.recommended && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-rh-green-light dark:bg-rh-green-dark/40 px-1.5 py-0.5 text-[10px] font-semibold text-rh-green-dark dark:text-rh-green">
                        <CircleCheck className="h-2.5 w-2.5" />
                        Recommended
                      </span>
                    )}
                  </div>
                  {model.description && (
                    <p className="text-xs text-muted-foreground mt-0.5">{model.description}</p>
                  )}
                </div>
                <div className="text-right ml-3 shrink-0">
                  <p className="text-sm font-bold tabular-nums text-foreground">{formatUSD(model.total_cost)}</p>
                </div>
              </div>
            ))}
          </div>
        )}

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Estimates based on average token usage per sample &middot; {formatUSD(estimate.cost_per_sample)}/sample
        </p>
      </CardContent>
    </Card>
  )
}
