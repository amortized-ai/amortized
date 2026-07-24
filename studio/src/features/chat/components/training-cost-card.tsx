import { Card, CardContent } from "@/components/ui/card"
import { Cpu } from "lucide-react"

interface TrainingModel {
  model_id: string
  label: string
  description: string
  gpu_type: string
  vram_gb: number
  estimated_time_minutes: number
  estimated_cost: number
  cost_per_gpu_hour: number
}

interface TrainingCostEstimate {
  num_samples: number
  num_epochs: number
  models: TrainingModel[]
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

interface TrainingCostCardProps {
  estimate: TrainingCostEstimate
}

export function TrainingCostCard({ estimate }: TrainingCostCardProps) {
  return (
    <Card className="border-rh-blue/30 dark:border-rh-blue-dark/50 bg-rh-blue-light/30 dark:bg-rh-blue-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <Cpu className="h-4 w-4 text-rh-blue dark:text-rh-blue" />
          <span className="text-sm font-semibold text-foreground">Training Cost Estimate</span>
          <span className="ml-auto text-xs text-muted-foreground">
            {estimate.num_samples.toLocaleString()} samples &middot; {estimate.num_epochs} epochs
          </span>
        </div>

        <div className="space-y-2">
          {estimate.models.map((model) => (
            <div
              key={model.model_id}
              className="flex items-center justify-between rounded-lg border border-rh-blue-light dark:border-rh-blue-dark/50 bg-background/60 dark:bg-rh-gray-90/30 px-3 py-2"
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground">{model.label}</p>
                <p className="text-xs text-muted-foreground">
                  {model.gpu_type} &middot; {model.vram_gb}GB VRAM &middot; {formatTime(model.estimated_time_minutes)}
                </p>
              </div>
              <div className="text-right ml-3">
                <p className="text-sm font-bold text-foreground">{formatUSD(model.estimated_cost)}</p>
                <p className="text-[10px] text-muted-foreground">{formatUSD(model.cost_per_gpu_hour)}/hr</p>
              </div>
            </div>
          ))}
        </div>

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Estimates based on single-GPU LoRA fine-tuning
        </p>
      </CardContent>
    </Card>
  )
}
