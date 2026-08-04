import { useState } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  CartesianGrid,
} from "recharts"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Skeleton } from "@/components/ui/skeleton"
import { ChartSkeleton } from "@/components/chart-skeleton"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from "@/components/ui/empty"
import { DollarSign } from "lucide-react"
import { useJobCostBreakdown } from "../api/use-jobs"

type TimeRange = "7d" | "30d" | "90d"

const costChartConfig = {
  training: {
    label: "Training",
    color: "var(--chart-1)",
  },
  sdg: {
    label: "SDG",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig

function formatCost(value: number): string {
  if (value === 0) return "$0"
  if (value < 0.01) return "<$0.01"
  if (value < 1000) return `$${value.toFixed(2)}`
  return `$${(value / 1000).toFixed(1)}k`
}

function CostTile({
  label,
  value,
  accentColor,
}: {
  label: string
  value: string
  accentColor?: string
}) {
  return (
    <div className="rounded-lg border bg-card p-3 flex flex-col gap-1">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span
        className="text-lg font-bold tabular-nums leading-tight"
        style={accentColor ? { color: accentColor } : undefined}
      >
        {value}
      </span>
    </div>
  )
}

function CostBreakdownSkeleton({ timeRange, onTimeRangeChange }: {
  timeRange: TimeRange
  onTimeRangeChange: (v: TimeRange) => void
}) {
  return (
    <Card className="animate-message-in" style={{ animationDelay: "150ms" }}>
      <CardHeader>
        <CardTitle>Cost Breakdown</CardTitle>
        <CardDescription>
          <Skeleton className="h-3 w-48" />
        </CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={timeRange}
            onValueChange={(v) => { if (v) onTimeRangeChange(v as TimeRange) }}
            variant="outline"
          >
            <ToggleGroupItem value="7d" className="h-8 px-2.5 text-xs">7d</ToggleGroupItem>
            <ToggleGroupItem value="30d" className="h-8 px-2.5 text-xs">30d</ToggleGroupItem>
            <ToggleGroupItem value="90d" className="h-8 px-2.5 text-xs">90d</ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent>
        <div aria-busy="true" role="status">
          <span className="sr-only">Loading cost breakdown...</span>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="rounded-lg border bg-card p-3 space-y-2">
                <Skeleton className="h-3 w-16" />
                <Skeleton className="h-6 w-20" />
              </div>
            ))}
          </div>
          <div className="mt-4">
            <ChartSkeleton />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function CostBreakdownChart() {
  const [timeRange, setTimeRange] = useState<TimeRange>("30d")
  const { data, isLoading, isError } = useJobCostBreakdown(timeRange)

  const rangeDays = { "7d": 7, "30d": 30, "90d": 90 }[timeRange]

  if (isLoading) {
    return <CostBreakdownSkeleton timeRange={timeRange} onTimeRangeChange={setTimeRange} />
  }

  if (isError) {
    return (
      <Card className="animate-message-in" style={{ animationDelay: "150ms" }}>
        <CardHeader>
          <CardTitle>Cost Breakdown</CardTitle>
          <CardDescription>Estimated compute costs</CardDescription>
          <CardAction>
            <ToggleGroup
              type="single"
              value={timeRange}
              onValueChange={(v) => { if (v) setTimeRange(v as TimeRange) }}
              variant="outline"
            >
              <ToggleGroupItem value="7d" className="h-8 px-2.5 text-xs">7d</ToggleGroupItem>
              <ToggleGroupItem value="30d" className="h-8 px-2.5 text-xs">30d</ToggleGroupItem>
              <ToggleGroupItem value="90d" className="h-8 px-2.5 text-xs">90d</ToggleGroupItem>
            </ToggleGroup>
          </CardAction>
        </CardHeader>
        <CardContent>
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon"><DollarSign /></EmptyMedia>
              <EmptyTitle>Cost tracking not available</EmptyTitle>
              <EmptyDescription>
                Cost estimates require completed jobs with duration data. Check that the backend API is running.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        </CardContent>
      </Card>
    )
  }

  const hasData = data && (data.total_cost > 0 || data.daily.some((d) => d.training > 0 || d.sdg > 0))

  if (!hasData) {
    return (
      <Card className="animate-message-in" style={{ animationDelay: "150ms" }}>
        <CardHeader>
          <CardTitle>Cost Breakdown</CardTitle>
          <CardDescription>
            {`No completed jobs in the last ${rangeDays} days`}
          </CardDescription>
          <CardAction>
            <ToggleGroup
              type="single"
              value={timeRange}
              onValueChange={(v) => { if (v) setTimeRange(v as TimeRange) }}
              variant="outline"
            >
              <ToggleGroupItem value="7d" className="h-8 px-2.5 text-xs">7d</ToggleGroupItem>
              <ToggleGroupItem value="30d" className="h-8 px-2.5 text-xs">30d</ToggleGroupItem>
              <ToggleGroupItem value="90d" className="h-8 px-2.5 text-xs">90d</ToggleGroupItem>
            </ToggleGroup>
          </CardAction>
        </CardHeader>
        <CardContent>
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon"><DollarSign /></EmptyMedia>
              <EmptyTitle>No cost data yet</EmptyTitle>
              <EmptyDescription>
                Costs appear after jobs finish running. Complete a training or SDG job to see your cost breakdown.
              </EmptyDescription>
            </EmptyHeader>
          </Empty>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="animate-message-in" style={{ animationDelay: "150ms" }}>
      <CardHeader>
        <CardTitle>Cost Breakdown</CardTitle>
        <CardDescription>
          {`Estimated compute costs over the last ${rangeDays} days`}
        </CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={timeRange}
            onValueChange={(v) => { if (v) setTimeRange(v as TimeRange) }}
            variant="outline"
          >
            <ToggleGroupItem value="7d" className="h-8 px-2.5 text-xs">7d</ToggleGroupItem>
            <ToggleGroupItem value="30d" className="h-8 px-2.5 text-xs">30d</ToggleGroupItem>
            <ToggleGroupItem value="90d" className="h-8 px-2.5 text-xs">90d</ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <CostTile label="Total Cost" value={formatCost(data.total_cost)} />
          <CostTile
            label="GPU Compute"
            value={formatCost(data.training_cost)}
            accentColor="var(--chart-1)"
          />
          <CostTile
            label="API Tokens"
            value={formatCost(data.sdg_cost)}
            accentColor="var(--chart-2)"
          />
        </div>
        <div className="mt-4">
          <ChartContainer config={costChartConfig} className="aspect-auto h-[200px] w-full">
            <BarChart data={data.daily} accessibilityLayer>
              <CartesianGrid vertical={false} className="stroke-muted" />
              <XAxis
                dataKey="date"
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                minTickGap={32}
                tickFormatter={(value: string) => {
                  const d = new Date(value + "T00:00:00")
                  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" })
                }}
              />
              <ChartTooltip
                content={
                  <ChartTooltipContent
                    labelFormatter={(value) =>
                      new Date(String(value) + "T00:00:00").toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })
                    }
                    formatter={(value) => `$${Number(value).toFixed(2)}`}
                  />
                }
              />
              <ChartLegend content={<ChartLegendContent />} />
              <Bar
                dataKey="sdg"
                stackId="cost"
                fill="var(--color-sdg)"
                fillOpacity={0.85}
                radius={[0, 0, 0, 0]}
              />
              <Bar
                dataKey="training"
                stackId="cost"
                fill="var(--color-training)"
                fillOpacity={0.85}
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ChartContainer>
        </div>
      </CardContent>
    </Card>
  )
}
