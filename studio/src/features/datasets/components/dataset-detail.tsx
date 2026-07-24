import { useState } from "react"
import { useParams, useNavigate } from "react-router"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowLeft, Copy, ExternalLink } from "lucide-react"
import { useDataset } from "../api/use-datasets"
import { TableSkeleton } from "@/components/table-skeleton"
import { JsonTreeViewer } from "@/components/json-tree-viewer"

export function DatasetDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { data: dataset, isLoading } = useDataset(id ?? null)
  const [copied, setCopied] = useState(false)

  if (isLoading) {
    return (
      <div className="p-4">
        <TableSkeleton columns={4} />
      </div>
    )
  }

  if (!dataset) {
    return (
      <div className="space-y-4">
        <Button variant="ghost" size="sm" onClick={() => void navigate("/datasets")}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back to Datasets
        </Button>
        <p className="text-sm text-muted-foreground">Dataset not found.</p>
      </div>
    )
  }

  function handleCopyId() {
    void navigator.clipboard.writeText(dataset!.run_id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="space-y-4" data-testid="dataset-detail">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => void navigate("/datasets")}
            className="mb-2"
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Datasets
          </Button>
          <h1 className="text-2xl font-bold">{dataset.name}</h1>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="font-mono text-xs">
              Run: {dataset.run_id}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={handleCopyId}
              aria-label="Copy run ID"
            >
              <Copy className="h-3 w-3" />
            </Button>
            {copied && (
              <span className="text-xs text-muted-foreground">Copied!</span>
            )}
          </div>
        </div>

        <Button variant="outline" size="sm" asChild>
          <a
            href={`/mlflow/#/experiments/${dataset.experiment_id}/runs/${dataset.run_id}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <ExternalLink className="h-4 w-4 mr-1" />
            View in MLflow
          </a>
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Parameters</CardTitle>
          </CardHeader>
          <CardContent>
            {Object.keys(dataset.params).length > 0 ? (
              <JsonTreeViewer data={dataset.params} collapsed={1} />
            ) : (
              <p className="text-sm text-muted-foreground">No parameters recorded.</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">Metrics</CardTitle>
          </CardHeader>
          <CardContent>
            {Object.keys(dataset.metrics).length > 0 ? (
              <dl className="space-y-2 text-sm">
                {Object.entries(dataset.metrics).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <dt className="text-muted-foreground">{key}</dt>
                    <dd className="font-mono">{value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-sm text-muted-foreground">No metrics recorded.</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Tags</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {Object.entries(dataset.tags).map(([key, value]) => (
              <Badge key={key} variant="outline" className="font-mono text-xs">
                {key}: {value}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
