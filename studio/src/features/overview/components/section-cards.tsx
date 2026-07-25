import { Link } from "react-router"
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Database, Box, Activity, TrendingUp } from "lucide-react"
import { useDatasets } from "@/features/datasets/api/use-datasets"
import { useModels } from "@/features/models/api/use-models"
import { useJobs } from "@/features/jobs/api/use-jobs"

interface MetricCardProps {
  title: string
  count: number | undefined
  isLoading: boolean
  description: string
  icon: React.ElementType
  path: string
}

function MetricCard({ title, count, isLoading, description, icon: Icon, path }: MetricCardProps) {
  return (
    <Link to={path}>
      <Card className="@container/card transition-colors hover:bg-accent/50">
        <CardHeader>
          <CardDescription>{title}</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {isLoading ? <Skeleton className="h-8 w-12" /> : (count ?? 0)}
          </CardTitle>
          <CardAction>
            <Badge variant="outline">
              <Icon className="size-3" />
              {!isLoading && (count ?? 0) > 0 ? (
                <>
                  <TrendingUp className="size-3" />
                  Active
                </>
              ) : (
                "—"
              )}
            </Badge>
          </CardAction>
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="text-muted-foreground">{description}</div>
        </CardFooter>
      </Card>
    </Link>
  )
}

export function SectionCards() {
  const { data: datasets, isLoading: datasetsLoading } = useDatasets()
  const { data: models, isLoading: modelsLoading } = useModels()
  const { data: jobs, isLoading: jobsLoading } = useJobs()

  const activeJobs = jobs?.filter((j) =>
    ["running", "queued", "provisioning"].includes(j.status)
  )

  return (
    <div className="grid grid-cols-1 gap-4 px-4 *:data-[slot=card]:shadow-xs lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-3">
      <MetricCard
        title="Total Datasets"
        count={datasets?.length}
        isLoading={datasetsLoading}
        description="Generated training datasets"
        icon={Database}
        path="/datasets"
      />
      <MetricCard
        title="Models"
        count={models?.length}
        isLoading={modelsLoading}
        description="Registered fine-tuned models"
        icon={Box}
        path="/models"
      />
      <MetricCard
        title="Active Jobs"
        count={activeJobs?.length}
        isLoading={jobsLoading}
        description="Training and SDG runs"
        icon={Activity}
        path="/jobs"
      />
    </div>
  )
}
