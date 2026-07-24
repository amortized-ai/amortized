import { Link } from "react-router"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Database, Box } from "lucide-react"
import { useDatasets } from "@/features/datasets/api/use-datasets"
import { useModels } from "@/features/models/api/use-models"

const ARTIFACTS = [
  { key: "datasets", label: "Datasets", icon: Database, path: "/datasets", cta: "Run an SDG recipe" },
  { key: "models", label: "Models", icon: Box, path: "/models", cta: "Train a model" },
] as const

export function ArtifactCounts() {
  const { data: datasets, isLoading: datasetsLoading } = useDatasets()
  const { data: models, isLoading: modelsLoading } = useModels()

  const counts: Record<string, number | undefined> = {
    datasets: datasets?.length,
    models: models?.length,
  }

  const loading: Record<string, boolean> = {
    datasets: datasetsLoading,
    models: modelsLoading,
  }

  return (
    <div className="grid grid-cols-2 gap-4">
      {ARTIFACTS.map((artifact) => (
        <Link key={artifact.key} to={artifact.path}>
          <Card className="transition-colors hover:bg-accent">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {artifact.label}
              </CardTitle>
              <artifact.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {loading[artifact.key] ? (
                  <Skeleton className="h-7 w-8" />
                ) : (
                  (counts[artifact.key] ?? 0)
                )}
              </div>
              {!loading[artifact.key] && (counts[artifact.key] ?? 0) === 0 && (
                <p className="text-xs text-primary mt-1">
                  {artifact.cta}
                </p>
              )}
            </CardContent>
          </Card>
        </Link>
      ))}
    </div>
  )
}
