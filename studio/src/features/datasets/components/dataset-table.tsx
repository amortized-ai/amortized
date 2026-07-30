import { type ColumnDef } from "@tanstack/react-table"
import { Database, Sparkles, Upload } from "lucide-react"
import { useMemo } from "react"
import { Badge } from "@/components/ui/badge"
import { EmptyState } from "@/components/empty-state"
import { DataTable } from "@/components/data-table"
import type { DatasetRecord } from "@/types/api"

interface DatasetTableProps {
  datasets: DatasetRecord[]
  page: number
  onPageChange: (page: number) => void
  onSelectDataset: (dataset: DatasetRecord) => void
}

export function DatasetTable({
  datasets,
  page,
  onPageChange,
  onSelectDataset,
}: DatasetTableProps) {
  const sorted = useMemo(
    () => [...datasets].sort((a, b) => b.created_at - a.created_at),
    [datasets],
  )

  const columns: ColumnDef<DatasetRecord, unknown>[] = useMemo(
    () => [
      {
        accessorKey: "name",
        header: "Name",
        cell: ({ row }) => {
          const isUpload = row.original.tags?.source === "upload"
          return (
            <div className="flex items-center gap-2">
              <span className="font-medium">{row.original.name}</span>
              {isUpload ? (
                <Badge
                  variant="secondary"
                  className="gap-0.5 py-0 text-[10px] bg-[#e0f0e8] text-[#1a7a42] border-0 dark:bg-[#0d331a]/40 dark:text-[#4aba78]"
                >
                  <Upload className="h-2.5 w-2.5" />
                  Upload
                </Badge>
              ) : (
                <Badge
                  variant="secondary"
                  className="gap-0.5 py-0 text-[10px] bg-[#ece6ff] text-[#5e40be] border-0 dark:bg-[#1b0d33]/40 dark:text-[#876fd4]"
                >
                  <Sparkles className="h-2.5 w-2.5" />
                  SDG
                </Badge>
              )}
            </div>
          )
        },
      },
      {
        accessorFn: (row) => row.metrics["num_samples_generated"] ?? null,
        id: "samples",
        header: "Samples",
        cell: ({ getValue }) => {
          const v = getValue() as number | null
          return (
            <span className="text-sm font-medium tabular-nums">
              {v != null ? v.toLocaleString() : "--"}
            </span>
          )
        },
      },
      {
        accessorFn: (row) => row.params["model"] ?? "--",
        id: "model",
        header: "Teacher Model",
        cell: ({ getValue }) => (
          <span className="text-sm text-muted-foreground truncate max-w-[200px] block">
            {getValue() as string}
          </span>
        ),
      },
      {
        accessorFn: (row) =>
          new Date(row.created_at).toLocaleString("en-US", {
            month: "short",
            day: "numeric",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit",
          }),
        id: "created_at",
        header: "Created",
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
      data={sorted}
      onRowClick={onSelectDataset}
      pagination={paginationState}
      onPaginationChange={(updater) => {
        const next = typeof updater === "function" ? updater(paginationState) : updater
        onPageChange(next.pageIndex)
      }}
      rowTestId={(ds) => `dataset-row-${ds.run_id}`}
      emptyState={
        <EmptyState
          icon={Database}
          title="No datasets yet"
          description="Run an SDG recipe to generate training data."
        />
      }
    />
  )
}
