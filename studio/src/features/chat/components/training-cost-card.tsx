import { Card, CardContent } from "@/components/ui/card"
import { Cpu, HardDrive, Clock } from "lucide-react"

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
          <span className="text-sm font-semibold text-foreground">Student Model Comparison</span>
          <span className="ml-auto text-xs text-muted-foreground">
            {estimate.num_samples.toLocaleString()} samples
          </span>
        </div>

        <div className="space-y-2">
          {estimate.models.map((model) => (
            <div
              key={model.model_id}
              className="flex items-center justify-between rounded-lg border border-rh-blue-light dark:border-rh-blue-dark/50 bg-background/60 dark:bg-rh-gray-90/30 px-3 py-2.5"
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground">{model.label}</p>
                <p className="text-xs text-muted-foreground mt-0.5">{model.description}</p>
              </div>
              <div className="flex items-center gap-4 ml-3 shrink-0">
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <HardDrive className="h-3 w-3" />
                  <span>{model.vram_gb}GB</span>
                </div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Cpu className="h-3 w-3" />
                  <span>{model.gpu_type}</span>
                </div>
                <div className="flex items-center gap-1 text-xs font-medium text-foreground">
                  <Clock className="h-3 w-3" />
                  <span>{formatTime(model.estimated_time_minutes)}</span>
                </div>
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
