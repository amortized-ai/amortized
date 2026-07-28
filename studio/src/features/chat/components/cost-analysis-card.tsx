import { Card, CardContent } from "@/components/ui/card"
import { TrendingDown } from "lucide-react"

interface CostEstimate {
  model_label: string
  num_samples: number
  cost: {
    input: number
    output: number
    total: number
  }
  comparison: {
    manual_labeling_total: number
    savings_amount: number
    savings_percent: number
  }
}

function formatUSD(value: number): string {
  const n = Number(value)
  if (n < 0.01 && n > 0) return "< $0.01"
  return `$${n.toFixed(2)}`
}

interface CostAnalysisCardProps {
  estimate: CostEstimate
  phase?: "sdg" | "training"
}

export function CostAnalysisCard({ estimate, phase = "sdg" }: CostAnalysisCardProps) {
  const { cost, comparison } = estimate

  const getPhaseLabel = () => {
    switch (phase) {
      case "training":
        return `Training with ${estimate.model_label}`
      default:
        return `SDG with ${estimate.model_label}`
    }
  }

  const getComparisonLabel = () => {
    return "Frontier model API"
  }

  return (
    <Card className="border-rh-green/30 dark:border-rh-green-dark/50 bg-rh-green-light/50 dark:bg-rh-green-dark/20 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <TrendingDown className="h-4 w-4 text-rh-green dark:text-rh-green" />
            <span className="text-sm font-semibold text-foreground">Cost Estimate</span>
          </div>
          <span className="inline-flex items-center rounded-full bg-gradient-to-r from-rh-green-light to-rh-green-light/50 dark:from-rh-green-dark/40 dark:to-rh-green-dark/20 px-2.5 py-0.5 text-xs font-semibold text-rh-green-dark dark:text-rh-green">
            Save {comparison.savings_percent.toFixed(0)}%
          </span>
        </div>

        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="space-y-1.5">
            <p className="font-medium text-muted-foreground">{getPhaseLabel()}</p>
            <p className="text-lg font-bold tabular-nums text-foreground">{formatUSD(cost.total)}</p>
            <p className="text-muted-foreground/70">
              {estimate.num_samples.toLocaleString()} samples
            </p>
          </div>

          <div className="space-y-1.5">
            <p className="font-medium text-muted-foreground">{getComparisonLabel()}</p>
            <p className="text-lg font-bold tabular-nums text-muted-foreground line-through decoration-1">
              {formatUSD(comparison.manual_labeling_total)}
            </p>
            <p className="font-medium text-rh-green dark:text-rh-green">
              You save {formatUSD(comparison.savings_amount)}
            </p>
          </div>
        </div>

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Estimates based on average token usage per sample
        </p>
      </CardContent>
    </Card>
  )
}
