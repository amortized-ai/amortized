import { Card, CardContent } from "@/components/ui/card"
import { Cpu } from "lucide-react"

interface VRAMEstimateData {
  model_size: string
  method: string
  num_gpus: number
  vram_per_gpu_gb: { low: number; expected: number; high: number }
  total_vram_gb: { low: number; expected: number; high: number }
}

function formatMethod(method: string): string {
  const labels: Record<string, string> = {
    sft: "Full SFT",
    lora: "LoRA",
    lora_sft: "LoRA",
    qlora: "QLoRA",
    qlora_sft: "QLoRA",
    osft: "OSFT",
  }
  return labels[method.toLowerCase()] ?? method
}

interface VRAMEstimateCardProps {
  data: VRAMEstimateData
}

export function VRAMEstimateCard({ data }: VRAMEstimateCardProps) {
  const v = data.vram_per_gpu_gb
  const isMultiGpu = data.num_gpus > 1

  return (
    <Card className="border-rh-blue/30 dark:border-rh-blue-dark/50 bg-rh-blue-light/30 dark:bg-rh-blue-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-rh-blue dark:text-rh-blue" />
            <span className="text-sm font-semibold text-foreground">VRAM Estimate</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="rounded-full bg-rh-blue-light dark:bg-rh-blue-dark/40 px-2 py-0.5 text-[10px] font-semibold text-rh-blue-dark dark:text-rh-blue">
              {data.model_size}
            </span>
            <span className="rounded-full bg-rh-blue-light dark:bg-rh-blue-dark/40 px-2 py-0.5 text-[10px] font-semibold text-rh-blue-dark dark:text-rh-blue">
              {formatMethod(data.method)}
            </span>
          </div>
        </div>

        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold tabular-nums text-foreground">
            {v.expected} GB
          </span>
          <span className="text-xs text-muted-foreground">
            per GPU
          </span>
        </div>

        <p className="text-xs text-muted-foreground mt-1">
          Range: {v.low}&ndash;{v.high} GB
        </p>

        {isMultiGpu && (
          <p className="text-xs text-muted-foreground mt-0.5">
            Total across {data.num_gpus} GPUs: {data.total_vram_gb.expected} GB
          </p>
        )}

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Estimates based on training-hub profiling formulas
        </p>
      </CardContent>
    </Card>
  )
}
