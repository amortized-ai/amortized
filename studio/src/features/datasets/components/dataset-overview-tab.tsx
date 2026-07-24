import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Clock, Database, Hash } from "lucide-react"
import type { DatasetRecord } from "@/types/api"

interface DatasetOverviewTabProps {
  dataset: DatasetRecord
}

export function DatasetOverviewTab({ dataset }: DatasetOverviewTabProps) {
  return (
    <div data-testid="overview-tab">
      <div className="space-y-6">
        <Card>
          <CardHeader className="bg-muted/50 rounded-t-lg">
            <CardTitle className="text-sm font-medium">Dataset Info</CardTitle>
          </CardHeader>
          <CardContent className="pt-4">
            <dl className="space-y-4 text-sm">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <dt className="text-muted-foreground mb-1">Run ID</dt>
                  <dd className="font-mono flex items-center gap-1">
                    <Hash className="h-3 w-3 text-muted-foreground" />
                    {dataset.run_id}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground mb-1">Model</dt>
                  <dd>{dataset.params["model"] ?? "—"}</dd>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <dt className="text-muted-foreground mb-1">Created</dt>
                  <dd className="flex items-center gap-1">
                    <Clock className="h-3 w-3 text-muted-foreground" />
                    {new Date(dataset.created_at).toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-muted-foreground mb-1">Samples</dt>
                  <dd className="flex items-center gap-1">
                    <Database className="h-3 w-3 text-muted-foreground" />
                    {dataset.metrics["num_samples_generated"] ?? "—"}
                  </dd>
                </div>
              </div>
            </dl>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
