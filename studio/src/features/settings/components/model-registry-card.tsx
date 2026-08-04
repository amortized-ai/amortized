import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { CardSkeleton } from "@/components/card-skeleton"
import { Box, ArrowRight } from "lucide-react"
import { Link } from "react-router"
import { useModels } from "@/features/models/api/use-models"
import { useConfig } from "@/features/settings/api/use-settings"
import { formatRelativeTime } from "@/lib/utils"

export function ModelRegistryCard() {
  const { data: models, isLoading, isError } = useModels()
  const { data: config } = useConfig()

  if (isLoading) {
    return <CardSkeleton />
  }

  const count = models?.length ?? 0
  const latest = models && models.length > 0
    ? [...models].sort((a, b) => b.created_at - a.created_at)[0]
    : null

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#daf2f2] text-[#147878] dark:bg-[#003333]/40 dark:text-[#37a3a3]">
            <Box className="h-3.5 w-3.5" />
          </div>
          <CardTitle className="text-sm">Model Registry</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        {isError ? (
          config?.mlflow_tracking_uri ? (
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Cannot reach the model registry.</p>
              <p className="text-xs text-muted-foreground">Check that MLflow is running.</p>
            </div>
          ) : (
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">Model registry is not yet configured.</p>
              <p className="text-xs text-muted-foreground">Set the MLflow tracking URI to enable model registration.</p>
            </div>
          )
        ) : count === 0 ? (
          <div className="space-y-3">
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">No models registered yet.</p>
              <p className="text-xs text-muted-foreground">Models are created by running training recipes.</p>
            </div>
            <div className="flex justify-center">
              <Button variant="outline" size="sm" asChild>
                <Link to="/recipes">Browse Recipes</Link>
              </Button>
            </div>
          </div>
        ) : (
          <div>
            <div className="flex items-center justify-between">
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-bold font-display">{count}</span>
                <span className="text-sm text-muted-foreground">
                  registered {count === 1 ? "model" : "models"}
                </span>
              </div>
              <Button variant="link" size="sm" asChild>
                <Link to="/models" className="gap-1">
                  Browse Models
                  <ArrowRight className="h-3 w-3" />
                </Link>
              </Button>
            </div>
            {latest && (
              <div className="flex items-center gap-3 mt-3 pt-3 border-t border-border/50">
                <div className="flex flex-col min-w-0 flex-1">
                  <span className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Latest</span>
                  <span className="text-sm font-medium truncate" title={latest.name}>{latest.name}</span>
                </div>
                <Badge variant="outline" className="font-mono text-xs shrink-0">
                  v{latest.version}
                </Badge>
                <span className="text-xs text-muted-foreground ml-auto shrink-0">
                  {formatRelativeTime(new Date(latest.created_at).toISOString())}
                </span>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
