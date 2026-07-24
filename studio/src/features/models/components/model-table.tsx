import { type ColumnDef } from "@tanstack/react-table"
import { Badge } from "@/components/ui/badge"
import { Box } from "lucide-react"
import { useMemo } from "react"
import { useNavigate } from "react-router"
import { EmptyState } from "@/components/empty-state"
import { DataTable } from "@/components/data-table"
import { formatDate } from "@/lib/utils"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import type { ModelRecord } from "@/types/api"

interface ModelTableProps {
  models: ModelRecord[]
  page: number
  onPageChange: (page: number) => void
  onSelectModel: (model: ModelRecord) => void
}

function ModelName({ model }: { model: ModelRecord }) {
  const customName = useEntityNamesStore((s) => s.names[model.name])
  return (
    <span className="font-medium text-sm">{customName || model.name}</span>
  )
}

const columns: ColumnDef<ModelRecord, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <ModelName model={row.original} />,
  },
  {
    accessorKey: "version",
    header: "Version",
    cell: ({ getValue }) => (
      <span className="text-sm text-muted-foreground">v{getValue() as string}</span>
    ),
  },
  {
    accessorFn: (row) => row.aliases.join(", ") || "—",
    id: "aliases",
    header: "Aliases",
    cell: ({ getValue }) => {
      const v = getValue() as string
      return v !== "—" ? (
        <Badge variant="outline" className="font-mono text-xs">{v}</Badge>
      ) : (
        <span className="text-sm text-muted-foreground">—</span>
      )
    },
  },
  {
    accessorFn: (row) => row.created_at,
    id: "created_at",
    header: "Created",
    cell: ({ row }) => (
      <span className="text-sm text-muted-foreground">
        {formatDate(new Date(row.original.created_at).toISOString())}
      </span>
    ),
  },
]

export function ModelTable({
  models,
  page,
  onPageChange,
  onSelectModel,
}: ModelTableProps) {
  const navigate = useNavigate()

  const sorted = useMemo(
    () => [...models].sort((a, b) => b.created_at - a.created_at),
    [models],
  )

  const paginationState = useMemo(
    () => ({ pageIndex: page, pageSize: 10 }),
    [page],
  )

  return (
    <DataTable
      columns={columns}
      data={sorted}
      onRowClick={onSelectModel}
      pagination={paginationState}
      onPaginationChange={(updater) => {
        const next = typeof updater === "function" ? updater(paginationState) : updater
        onPageChange(next.pageIndex)
      }}
      rowTestId={(model) => `model-row-${model.name}`}
      emptyState={
        <EmptyState
          icon={Box}
          title="No models yet"
          description="Trained models will appear here after running a training recipe."
          actionLabel="Create a Recipe"
          onAction={() => navigate("/recipes")}
        />
      }
    />
  )
}
