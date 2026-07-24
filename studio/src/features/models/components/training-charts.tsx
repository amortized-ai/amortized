import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { ChartSkeleton } from "@/components/chart-skeleton"
import type { TrainingMetric } from "@/types/api"

const lossConfig = {
  loss: {
    label: "Loss",
    color: "var(--chart-1)",
  },
} satisfies ChartConfig

const gradNormConfig = {
  grad_norm: {
    label: "Grad Norm",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig

interface TrainingChartsProps {
  metrics: TrainingMetric[]
  isLoading: boolean
}

export function TrainingCharts({ metrics, isLoading }: TrainingChartsProps) {
  if (isLoading) {
    return <ChartSkeleton />
  }

  if (metrics.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No training metrics available.
      </div>
    )
  }

  const hasGradNorm = metrics.some((m) => m.grad_norm != null)

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Training Loss</CardTitle>
        </CardHeader>
        <CardContent>
          <ChartContainer config={lossConfig} className="min-h-[240px] w-full">
            <LineChart data={metrics} accessibilityLayer>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="step"
                label={{ value: "Step", position: "insideBottom", offset: -5 }}
                className="text-xs"
              />
              <YAxis
                label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                className="text-xs"
              />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="var(--color-loss)"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {hasGradNorm && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Gradient Norm</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer config={gradNormConfig} className="min-h-[240px] w-full">
              <LineChart data={metrics} accessibilityLayer>
                <CartesianGrid
                  strokeDasharray="3 3"
                  className="stroke-muted"
                />
                <XAxis
                  dataKey="step"
                  label={{
                    value: "Step",
                    position: "insideBottom",
                    offset: -5,
                  }}
                  className="text-xs"
                />
                <YAxis
                  label={{
                    value: "Grad Norm",
                    angle: -90,
                    position: "insideLeft",
                  }}
                  className="text-xs"
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Line
                  type="monotone"
                  dataKey="grad_norm"
                  stroke="var(--color-grad_norm)"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
