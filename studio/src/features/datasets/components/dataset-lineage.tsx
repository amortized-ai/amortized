import { Link } from "react-router"

interface DatasetLineageProps {
  parentJobId: string | null
}

export function DatasetLineage({ parentJobId }: DatasetLineageProps) {
  if (!parentJobId) {
    return (
      <p className="text-sm text-muted-foreground">
        No parent job linked.
      </p>
    )
  }

  return (
    <div className="space-y-2" data-testid="dataset-lineage">
      <Link
        to="/jobs"
        className="block text-sm text-primary hover:underline"
      >
        Parent Job: {parentJobId}
      </Link>
    </div>
  )
}
