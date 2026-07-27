import { Badge } from "@/components/ui/badge"
import type { JobType } from "@/types/api"

const TYPE_LABELS: Record<JobType, string> = {
  training: "Training",
  sdg: "SDG",
  eval: "Eval",
}

export function JobTypeBadge({ type }: { type: JobType }) {
  return (
    <Badge variant="secondary" data-job-type={type}>
      {TYPE_LABELS[type]}
    </Badge>
  )
}
