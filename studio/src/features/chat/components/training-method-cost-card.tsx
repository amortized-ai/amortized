import { Card, CardContent } from "@/components/ui/card"
import { Layers, CircleCheck } from "lucide-react"
import { cn } from "@/lib/utils"

interface TrainingMethod {
  method: string
  label: string
  description: string
  gpu_type: string
  vram_gb: number
  estimated_time_minutes: number
  estimated_cost: number
  relative_time: string
  recommended: boolean
}

interface TrainingMethodCostEstimate {
  model_id: string
  model_label: string
  num_samples: number
  num_epochs: number
  methods: TrainingMethod[]
}

function formatUSD(value: number): string {
  const n = Number(value)
  if (n < 0.01 && n > 0) return "< $0.01"
  return `$${n.toFixed(2)}`
}

function formatTime(minutes: number): string {
  if (minutes < 1) return "< 1 min"
  if (minutes < 60) return `~${Math.round(minutes)} min`
  const hours = Math.floor(minutes / 60)
  const mins = Math.round(minutes % 60)
  return mins > 0 ? `~${hours}h ${mins}m` : `~${hours}h`
}

interface TrainingMethodCostCardProps {
  estimate: TrainingMethodCostEstimate
}

export function TrainingMethodCostCard({ estimate }: TrainingMethodCostCardProps) {
  return (
    <Card className="border-rh-blue/30 dark:border-rh-blue-dark/50 bg-rh-blue-light/30 dark:bg-rh-blue-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <Layers className="h-4 w-4 text-rh-blue dark:text-rh-blue" />
          <span className="text-sm font-semibold text-foreground">Training Method Comparison</span>
          <span className="ml-auto text-xs text-muted-foreground">
            {estimate.model_label} &middot; {estimate.num_samples.toLocaleString()} samples
          </span>
        </div>

        <div className="space-y-2">
          {estimate.methods.map((method) => (
            <div
              key={method.method}
              className={cn(
                "flex items-center justify-between rounded-lg border px-3 py-2.5",
                method.recommended
                  ? "border-rh-green/30 dark:border-rh-green-dark/50 bg-rh-green-light/50 dark:bg-rh-green-dark/20"
                  : "border-rh-blue-light dark:border-rh-blue-dark/30 bg-background/60 dark:bg-rh-gray-90/30",
              )}
            >
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium text-foreground">{method.label}</p>
                  {method.recommended && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-rh-green-light dark:bg-rh-green-dark/40 px-1.5 py-0.5 text-[10px] font-semibold text-rh-green-dark dark:text-rh-green">
                      <CircleCheck className="h-2.5 w-2.5" />
                      Recommended
                    </span>
                  )}
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">{method.description}</p>
                <p className="text-xs text-muted-foreground/70 mt-0.5">
                  {method.gpu_type} &middot; {method.vram_gb}GB VRAM &middot; {formatTime(method.estimated_time_minutes)} &middot; {method.relative_time}
                </p>
              </div>
              <div className="text-right ml-3 shrink-0">
                <p className="text-sm font-bold tabular-nums text-foreground">{formatUSD(method.estimated_cost)}</p>
              </div>
            </div>
          ))}
        </div>

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Estimates based on {estimate.num_epochs} epochs &middot; costs vary by provider
        </p>
      </CardContent>
    </Card>
  )
}
