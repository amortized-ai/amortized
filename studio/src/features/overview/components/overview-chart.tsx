import { useState, useMemo } from "react"
import {
  Area,
  AreaChart,
  CartesianGrid,
  XAxis,
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
  type ChartConfig,
} from "@/components/ui/chart"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { useJobs } from "@/features/jobs/api/use-jobs"

const chartConfig = {
  training: {
    label: "Training",
    color: "var(--chart-1)",
  },
  evaluation: {
    label: "Evaluation",
    color: "var(--chart-2)",
  },
} satisfies ChartConfig

type TimeRange = "7d" | "30d" | "90d"

function generateDateRange(days: number) {
  const dates: string[] = []
  const now = new Date()
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(now)
    d.setDate(d.getDate() - i)
    dates.push(d.toISOString().slice(0, 10))
  }
  return dates
}

const RANGE_DAYS: Record<TimeRange, number> = {
  "7d": 7,
  "30d": 30,
  "90d": 90,
}

export function OverviewChart() {
  const [timeRange, setTimeRange] = useState<TimeRange>("30d")
  const { data: jobs = [] } = useJobs()

  const chartData = useMemo(() => {
    const days = RANGE_DAYS[timeRange]
    const dates = generateDateRange(days)

    return dates.map((date) => {
      const dayJobs = jobs.filter((j) => j.created_at.slice(0, 10) === date)
      return {
        date,
        training: dayJobs.filter((j) => j.type === "training" || j.type === "sdg").length,
        evaluation: dayJobs.filter((j) => j.type === "eval").length,
      }
    })
  }, [jobs, timeRange])

  const hasData = chartData.some((d) => d.training + d.evaluation > 0)

  // When there is no real job data, show sample data so the chart is not empty
  const displayData = useMemo(() => {
    if (hasData) return chartData

    const days = RANGE_DAYS[timeRange]
    const dates = generateDateRange(days)
    return dates.map((date, i) => ({
      date,
      training: Math.max(0, Math.round(Math.sin(i * 0.4) * 2 + 2 + ((i * 7 + 3) % 5) * 0.3)),
      evaluation: Math.max(0, Math.round(Math.cos(i * 0.3) * 1.5 + 1.5 + ((i * 11 + 7) % 4) * 0.25)),
    }))
  }, [hasData, chartData, timeRange])

  return (
    <Card className="@container/card">
      <CardHeader>
        <CardTitle>Job Activity</CardTitle>
        <CardDescription>
          {hasData
            ? `Training and evaluation jobs over the last ${RANGE_DAYS[timeRange]} days`
            : "Sample data — run some jobs to see real activity"}
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
        <ChartContainer config={chartConfig} className="aspect-auto h-[250px] w-full">
          <AreaChart data={displayData}>
            <defs>
              <linearGradient id="fillTraining" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-training)" stopOpacity={0.8} />
                <stop offset="95%" stopColor="var(--color-training)" stopOpacity={0.1} />
              </linearGradient>
              <linearGradient id="fillEvaluation" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--color-evaluation)" stopOpacity={0.8} />
                <stop offset="95%" stopColor="var(--color-evaluation)" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid vertical={false} />
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
                  indicator="dot"
                  labelFormatter={(value) => {
                    return new Date(String(value) + "T00:00:00").toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })
                  }}
                />
              }
            />
            <Area
              dataKey="evaluation"
              type="natural"
              fill="url(#fillEvaluation)"
              stroke="var(--color-evaluation)"
              stackId="a"
            />
            <Area
              dataKey="training"
              type="natural"
              fill="url(#fillTraining)"
              stroke="var(--color-training)"
              stackId="a"
            />
          </AreaChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
