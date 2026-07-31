import { Card, CardContent } from "@/components/ui/card"
import { Cpu } from "lucide-react"
import { cn } from "@/lib/utils"

interface VRAMEstimateItem {
  model_size: string
  method: string
  vram_per_gpu_gb: number
  vram_range: string
}

interface VRAMEstimateData {
  estimates: VRAMEstimateItem[]
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
  const lowest = data.estimates.length > 0
    ? data.estimates.reduce((min, e) => e.vram_per_gpu_gb < min.vram_per_gpu_gb ? e : min, data.estimates[0]!)
    : null

  return (
    <Card className="border-rh-blue/30 dark:border-rh-blue-dark/50 bg-rh-blue-light/30 dark:bg-rh-blue-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-rh-blue dark:text-rh-blue" />
            <span className="text-sm font-semibold text-foreground">VRAM Estimate</span>
          </div>
          <span className="text-xs text-muted-foreground">per GPU</span>
        </div>

        <div className="space-y-1.5">
          {data.estimates.map((est, i) => {
            const isLowest = lowest !== null && est === lowest && data.estimates.length > 1

            return (
              <div
                key={`${est.model_size}-${est.method}-${i}`}
                className={cn(
                  "flex items-center justify-between rounded-lg border px-3 py-2",
                  isLowest
                    ? "border-rh-green/30 dark:border-rh-green-dark/50 bg-rh-green-light/50 dark:bg-rh-green-dark/20"
                    : "border-rh-blue-light dark:border-rh-blue-dark/30 bg-background/60 dark:bg-rh-gray-90/30",
                )}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-foreground">{est.model_size}</span>
                    <span className="rounded-full bg-rh-blue-light dark:bg-rh-blue-dark/40 px-1.5 py-0.5 text-[10px] font-semibold text-rh-blue-dark dark:text-rh-blue">
                      {formatMethod(est.method)}
                    </span>
                    {isLowest && (
                      <span className="rounded-full bg-rh-green-light dark:bg-rh-green-dark/40 px-1.5 py-0.5 text-[10px] font-semibold text-rh-green-dark dark:text-rh-green">
                        Lowest
                      </span>
                    )}
                  </div>
                  {est.vram_range && (
                    <p className="text-[10px] text-muted-foreground mt-0.5">Range: {est.vram_range}</p>
                  )}
                </div>
                <span className="text-sm font-bold tabular-nums text-foreground ml-3 shrink-0">
                  {est.vram_per_gpu_gb} GB
                </span>
              </div>
            )
          })}
        </div>

        <p className="mt-3 text-[10px] text-muted-foreground/50">
          Estimates based on training-hub profiling formulas
        </p>
      </CardContent>
    </Card>
  )
}
