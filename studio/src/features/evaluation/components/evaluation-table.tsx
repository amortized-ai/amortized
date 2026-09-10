import { type ColumnDef } from "@tanstack/react-table"
import { ClipboardCheck } from "lucide-react"
import { useMemo } from "react"
import { EmptyState } from "@/components/empty-state"
import { DataTable } from "@/components/data-table"
import type { EvaluationGroup } from "@/lib/api-client"

interface EvaluationTableProps {
  groups: EvaluationGroup[]
  page: number
  onPageChange: (page: number) => void
  onSelectGroup: (group: EvaluationGroup) => void
}

export function EvaluationTable({
  groups,
  page,
  onPageChange,
  onSelectGroup,
}: EvaluationTableProps) {
  const columns: ColumnDef<EvaluationGroup, unknown>[] = useMemo(
    () => [
      {
        accessorFn: (row) => row.dataset.name,
        id: "dataset",
        header: "Dataset",
        size: 280,
        minSize: 150,
        maxSize: 500,
        cell: ({ row }) => (
          <span className="font-medium truncate block">
            {row.original.dataset.name}
          </span>
        ),
      },
      {
        accessorFn: (row) => row.evals.length,
        id: "models",
        header: "Models",
        size: 90,
        minSize: 70,
        maxSize: 130,
        cell: ({ getValue }) => (
          <span className="text-sm font-medium tabular-nums">
            {getValue() as number}
          </span>
        ),
      },
      {
        accessorFn: (row) => row.metric_names.length,
        id: "metrics",
        header: "Metrics",
        size: 90,
        minSize: 70,
        maxSize: 130,
        cell: ({ getValue }) => (
          <span className="text-sm font-medium tabular-nums">
            {getValue() as number}
          </span>
        ),
      },
      {
        accessorFn: (row) =>
          row.evals
            .filter((e) => e.status === "succeeded")
            .map((e) => e.model)
            .join(", "),
        id: "models_list",
        header: "Evaluated Models",
        size: 280,
        minSize: 150,
        maxSize: 500,
        cell: ({ getValue }) => (
          <span className="text-sm text-muted-foreground truncate block">
            {(getValue() as string) || "--"}
          </span>
        ),
      },
      {
        accessorFn: (row) => formatDate(row.latest_created_at),
        id: "created",
        header: "Latest Eval",
        size: 170,
        minSize: 140,
        maxSize: 250,
        cell: ({ getValue }) => (
          <span className="text-sm text-muted-foreground whitespace-nowrap">
            {getValue() as string}
          </span>
        ),
      },
    ],
    [],
  )

  const paginationState = useMemo(
    () => ({ pageIndex: page, pageSize: 10 }),
    [page],
  )

  return (
    <DataTable
      columns={columns}
      data={groups}
      onRowClick={onSelectGroup}
      storageKey="evaluations-column-sizes"
      pagination={paginationState}
      onPaginationChange={(updater) => {
        const next = typeof updater === "function" ? updater(paginationState) : updater
        onPageChange(next.pageIndex)
      }}
      rowTestId={(g) => `evaluation-row-${g.id}`}
      emptyState={
        <EmptyState
          icon={ClipboardCheck}
          title="No evaluations yet"
          description="Ask Morty to evaluate a model — results and model comparisons appear here."
        />
      }
    />
  )
}

function formatDate(value: string | null): string {
  if (!value) return "--"
  return new Date(value).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  })
}
