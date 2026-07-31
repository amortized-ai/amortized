import { Card, CardContent } from "@/components/ui/card"
import { DollarSign } from "lucide-react"
import { cn } from "@/lib/utils"

interface ModelPricingModel {
  model_id: string
  name: string
  prompt_cost_per_1m: number
  completion_cost_per_1m: number
  context_length: number
}

interface ModelPricingData {
  query: string
  models: ModelPricingModel[]
}

function formatCost(value: number): string {
  if (value === 0) return "Free"
  if (value < 0.01) return `$${value.toFixed(4)}`
  if (value < 1) return `$${value.toFixed(2)}`
  return `$${value.toFixed(2)}`
}

function formatContext(length: number): string {
  if (length >= 1_000_000) return `${(length / 1_000_000).toFixed(0)}M`
  if (length >= 1_000) return `${(length / 1_000).toFixed(0)}K`
  return `${length}`
}

interface ModelPricingCardProps {
  data: ModelPricingData
}

export function ModelPricingCard({ data }: ModelPricingCardProps) {
  const cheapest = data.models.length > 0
    ? data.models.reduce((min, m) =>
        m.prompt_cost_per_1m < min.prompt_cost_per_1m ? m : min, data.models[0]!)
    : null

  return (
    <Card className="border-rh-purple/30 dark:border-rh-purple-dark/50 bg-rh-purple-light/30 dark:bg-rh-purple-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <DollarSign className="h-4 w-4 text-rh-purple dark:text-rh-purple" />
            <span className="text-sm font-semibold text-foreground">Model Pricing</span>
          </div>
          <span className="text-xs text-muted-foreground">per 1M tokens</span>
        </div>

        <div className="space-y-1.5">
          <div className="grid grid-cols-[1fr_auto_auto_auto] gap-2 px-2 text-[10px] font-medium text-muted-foreground/70 uppercase tracking-wider">
            <span>Model</span>
            <span className="text-right w-16">Input</span>
            <span className="text-right w-16">Output</span>
            <span className="text-right w-12">Ctx</span>
          </div>
          {data.models.map((model) => {
            const isCheapest = cheapest !== null && model.model_id === cheapest.model_id

            return (
              <div
                key={model.model_id}
                className={cn(
                  "grid grid-cols-[1fr_auto_auto_auto] gap-2 items-center rounded-lg border px-2 py-2",
                  isCheapest
                    ? "border-rh-green/30 dark:border-rh-green-dark/50 bg-rh-green-light/50 dark:bg-rh-green-dark/20"
                    : "border-rh-purple-light dark:border-rh-purple-dark/30 bg-background/60 dark:bg-rh-gray-90/30",
                )}
              >
                <div className="min-w-0">
                  <span className="text-xs font-medium text-foreground truncate block">{model.name}</span>
                </div>
                <span className="text-xs font-bold tabular-nums text-foreground text-right w-16">
                  {formatCost(model.prompt_cost_per_1m)}
                </span>
                <span className="text-xs font-bold tabular-nums text-foreground text-right w-16">
                  {formatCost(model.completion_cost_per_1m)}
                </span>
                <span className="text-[10px] tabular-nums text-muted-foreground text-right w-12">
                  {formatContext(model.context_length)}
                </span>
              </div>
            )
          })}
        </div>

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Prices from OpenRouter
        </p>
      </CardContent>
    </Card>
  )
}
