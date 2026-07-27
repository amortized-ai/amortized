import { type ColumnDef } from "@tanstack/react-table"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Briefcase, Copy, Check } from "lucide-react"
import { useState, useMemo } from "react"
import { useNavigate } from "react-router"
import { EmptyState } from "@/components/empty-state"
import { DataTable } from "@/components/data-table"
import { StatusBadge } from "./status-badge"
import { JobTypeBadge } from "./job-type-badge"
import { formatDuration } from "../lib/format"
import { formatDate } from "@/lib/utils"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import type { Job } from "@/types/api"

interface JobTableProps {
  jobs: Job[]
  page: number
  onPageChange: (page: number) => void
  onSelectJob: (job: Job) => void
}

function JobName({ job }: { job: Job }) {
  const customName = useEntityNamesStore((s) => s.names[job.id])
  return (
    <span className="font-medium text-sm">
      {customName || (job.metadata?.name as string) || job.id}
    </span>
  )
}

const columns: ColumnDef<Job, unknown>[] = [
  {
    accessorFn: (row) => (row.metadata?.name as string) || row.id,
    id: "name",
    header: "Name",
    cell: ({ row }) => <JobName job={row.original} />,
  },
  {
    accessorKey: "type",
    header: "Type",
    cell: ({ row }) => <JobTypeBadge type={row.original.type} />,
  },
  {
    accessorKey: "status",
    header: "Status",
    cell: ({ row }) => <StatusBadge status={row.original.status} />,
  },
  {
    accessorKey: "id",
    header: "ID",
    enableSorting: false,
    cell: ({ row }) => <CopyableId id={row.original.id} />,
  },
  {
    id: "elapsed",
    header: "Elapsed",
    accessorFn: (row) => row.started_at,
    cell: ({ row }) => (
      <span className="text-sm">
        {formatDuration(row.original.started_at, row.original.completed_at)}
      </span>
    ),
  },
  {
    accessorKey: "created_at",
    header: "Created",
    cell: ({ getValue }) => (
      <span className="text-sm text-muted-foreground">
        {formatDate(getValue() as string, { includeTime: true })}
      </span>
    ),
  },
]

export function JobTable({ jobs, page, onPageChange, onSelectJob }: JobTableProps) {
  const navigate = useNavigate()

  const paginationState = useMemo(
    () => ({ pageIndex: page, pageSize: 10 }),
    [page],
  )

  return (
    <DataTable
      columns={columns}
      data={jobs}
      onRowClick={onSelectJob}
      pagination={paginationState}
      onPaginationChange={(updater) => {
        const next = typeof updater === "function" ? updater(paginationState) : updater
        onPageChange(next.pageIndex)
      }}
      rowTestId={(job) => `job-row-${job.id}`}
      emptyState={
        <EmptyState
          icon={Briefcase}
          title="No jobs yet"
          description="Jobs appear here when you run training, SDG, eval, or serve tasks."
          actionLabel="Go to Chat"
          onAction={() => navigate("/chat")}
        />
      }
    />
  )
}

function CopyableId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false)
  const short = id.length > 8 ? `${id.slice(0, 8)}…` : id

  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation()
    void navigator.clipboard.writeText(id)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={handleCopy}
          className="inline-flex items-center gap-1 font-mono text-xs text-muted-foreground hover:text-foreground transition-colors"
          aria-label={`Copy job ID ${id}`}
        >
          {short}
          {copied ? (
            <Check className="h-3 w-3 text-rh-green dark:text-rh-green" />
          ) : (
            <Copy className="h-3 w-3" />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent>{id}</TooltipContent>
    </Tooltip>
  )
}
