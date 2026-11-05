import { Badge } from "@/components/ui/badge"
import type { JobType } from "@/types/api"
import { JOB_TYPE_LABELS } from "../lib/format"

export function JobTypeBadge({ type }: { type: JobType }) {
  return (
    <Badge variant="secondary" data-job-type={type}>
      {JOB_TYPE_LABELS[type]}
    </Badge>
  )
}
