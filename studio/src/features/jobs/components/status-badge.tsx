import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { statusVariants } from "@/lib/status-variants"
import type { JobStatus } from "@/types/api"

const STATUS_CONFIG: Record<
  JobStatus,
  { label: string; className: string }
> = {
  queued: {
    label: "Queued",
    className: statusVariants.neutral,
  },
  provisioning: {
    label: "Provisioning",
    className: statusVariants.info,
  },
  running: {
    label: "Running",
    className: `${statusVariants.info} animate-pulse`,
  },
  succeeded: {
    label: "Succeeded",
    className: statusVariants.success,
  },
  failed: {
    label: "Failed",
    className: statusVariants.error,
  },
  cancelled: {
    label: "Cancelled",
    className: statusVariants.warning,
  },
}

export function StatusBadge({ status }: { status: JobStatus }) {
  const config = STATUS_CONFIG[status]

  return (
    <Badge
      variant="outline"
      className={cn("font-medium", config.className)}
      data-status={status}
    >
      {config.label}
    </Badge>
  )
}
