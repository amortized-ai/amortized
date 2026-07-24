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
import { Skeleton } from "@/components/ui/skeleton"
import type { AllMetricsData, MetricHistory } from "../api/use-jobs"

/* ------------------------------------------------------------------ */
/*  Chart configs using rh-* color tokens                              */
/* ------------------------------------------------------------------ */

const lossConfig = {
  value: { label: "Loss", color: "var(--color-rh-red)" },
} satisfies ChartConfig

const gradNormConfig = {
  value: { label: "Grad Norm", color: "var(--color-rh-blue)" },
} satisfies ChartConfig

const lrConfig = {
  value: { label: "Learning Rate", color: "var(--color-rh-teal)" },
} satisfies ChartConfig

/* ------------------------------------------------------------------ */
/*  Value formatters                                                   */
/* ------------------------------------------------------------------ */

function fmtLoss(v: number | undefined): string {
  if (v == null) return "--"
  return v < 0.001 ? v.toExponential(2) : v.toFixed(4)
}

function fmtLr(v: number | undefined): string {
  if (v == null) return "--"
  return v.toExponential(2)
}

function fmtDuration(seconds: number | undefined): string {
  if (seconds == null) return "--"
  const s = Math.round(seconds)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const rem = s % 60
  if (m < 60) return `${m}m ${rem}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function fmtEpoch(v: number | undefined): string {
  if (v == null) return "--"
  return Number.isInteger(v) ? v.toString() : v.toFixed(1)
}

/* ------------------------------------------------------------------ */
/*  Summary card                                                       */
/* ------------------------------------------------------------------ */

function SummaryCard({
  label,
  value,
  color,
}: {
  label: string
  value: string
  color?: string
}) {
  return (
    <div className="rounded-lg border bg-card p-3 flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span
        className="text-lg font-bold tabular-nums leading-tight"
        style={color ? { color } : undefined}
      >
        {value}
      </span>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Single metric chart (reusable)                                     */
/* ------------------------------------------------------------------ */

function MetricChart({
  title,
  data,
  config,
  yLabel,
  formatValue,
}: {
  title: string
  data: MetricHistory[] | undefined
  config: ChartConfig
  yLabel: string
  formatValue?: (v: number) => string
}) {
  if (!data || data.length === 0) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
            No data available
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="min-h-[200px] w-full">
          <LineChart data={data} accessibilityLayer>
            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
            <XAxis
              dataKey="step"
              label={{ value: "Step", position: "insideBottom", offset: -5 }}
              className="text-xs"
              tickFormatter={(v: number) =>
                v >= 1000 ? `${(v / 1000).toFixed(0)}k` : String(v)
              }
            />
            <YAxis
              label={{ value: yLabel, angle: -90, position: "insideLeft" }}
              className="text-xs"
              tickFormatter={formatValue}
            />
            <ChartTooltip content={<ChartTooltipContent />} />
            <Line
              type="monotone"
              dataKey="value"
              stroke="var(--color-value)"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}

/* ------------------------------------------------------------------ */
/*  Loading skeleton                                                   */
/* ------------------------------------------------------------------ */

function MetricsSkeleton() {
  return (
    <div className="space-y-4" aria-busy="true" role="status">
      {/* Summary card skeletons */}
      <div className="grid grid-cols-4 gap-3">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="rounded-lg border bg-card p-3 space-y-2">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-6 w-20" />
          </div>
        ))}
      </div>
      {/* Chart skeletons */}
      <div className="grid gap-4 md:grid-cols-2">
        <Skeleton className="h-[280px] rounded-xl" />
        <Skeleton className="h-[280px] rounded-xl" />
        <Skeleton className="h-[280px] rounded-xl" />
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ */
/*  Main exported component                                            */
/* ------------------------------------------------------------------ */

interface TrainingMetricsChartProps {
  data: AllMetricsData | undefined
  isLoading: boolean
}

export function TrainingMetricsChart({ data, isLoading }: TrainingMetricsChartProps) {
  if (isLoading) {
    return <MetricsSkeleton />
  }

  if (!data) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-muted-foreground">
        No training metrics available.
      </div>
    )
  }

  const { histories, finalValues, params } = data

  const lastLoss = finalValues["loss"] ?? finalValues["train_loss"]
  const lastLr = finalValues["learning_rate"] ?? (params["learning_rate"] ? Number(params["learning_rate"]) : undefined)
  const trainRuntime = finalValues["train_runtime"]
  const lastEpoch = finalValues["epoch"]
  const lossHistory = histories["loss"] ?? histories["train_loss"]

  return (
    <div className="space-y-4">
      {/* ---- Summary cards ---- */}
      <div className="grid grid-cols-4 gap-3">
        <SummaryCard
          label="Final Loss"
          value={fmtLoss(lastLoss)}
          color="var(--color-rh-red)"
        />
        <SummaryCard
          label="Learning Rate"
          value={fmtLr(lastLr)}
          color="var(--color-rh-teal)"
        />
        <SummaryCard
          label="Training Time"
          value={fmtDuration(trainRuntime)}
        />
        <SummaryCard
          label="Epochs"
          value={fmtEpoch(lastEpoch)}
        />
      </div>

      {/* ---- Charts grid ---- */}
      <div className="grid gap-4 md:grid-cols-2">
        <MetricChart
          title="Training Loss"
          data={lossHistory}
          config={lossConfig}
          yLabel="Loss"
        />
        <MetricChart
          title="Gradient Norm"
          data={histories["grad_norm"]}
          config={gradNormConfig}
          yLabel="Grad Norm"
        />
        <MetricChart
          title="Learning Rate Schedule"
          data={histories["learning_rate"]}
          config={lrConfig}
          yLabel="LR"
          formatValue={(v: number) => v.toExponential(1)}
        />
      </div>
    </div>
  )
}
