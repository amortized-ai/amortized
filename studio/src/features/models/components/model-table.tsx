import { type ColumnDef } from "@tanstack/react-table"
import { Box } from "lucide-react"
import { useMemo } from "react"
import { useNavigate } from "react-router"
import { EmptyState } from "@/components/empty-state"
import { DataTable } from "@/components/data-table"
import { formatDate } from "@/lib/utils"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import { EditableTitle } from "@/components/editable-title"
import { setMlflowRegisteredModelTag } from "@/lib/api-client"
import { useQueryClient } from "@tanstack/react-query"
import type { ModelRecord } from "@/types/api"

interface ModelTableProps {
  models: ModelRecord[]
  page: number
  onPageChange: (page: number) => void
  onSelectModel: (model: ModelRecord) => void
}

function ModelName({ model }: { model: ModelRecord }) {
  const customName = useEntityNamesStore((s) => s.names[model.name])
  const displayName =
    model.tags?.model_display_name || customName || model.name
  return (
    <span className="font-medium text-sm" title={model.name}>{displayName}</span>
  )
}

function TopicCell({ model }: { model: ModelRecord }) {
  const queryClient = useQueryClient()
  const topic = model.tags?.model_topic ?? ""

  return (
    <div className="min-w-0" onClick={(e) => e.stopPropagation()} onKeyDown={(e) => e.stopPropagation()}>
      <EditableTitle
        value={topic || "Add topic..."}
        className={`text-sm ${!topic ? "text-muted-foreground/50 italic" : ""}`}
        onSave={async (newTopic) => {
          await setMlflowRegisteredModelTag(model.name, "model_topic", newTopic)
          queryClient.invalidateQueries({ queryKey: ["mlflow", "models"] })
        }}
      />
    </div>
  )
}

function BaseModelCell({ model }: { model: ModelRecord }) {
  const parts = model.name.split("-")
  if (parts.length >= 2) {
    const algo = parts[parts.length - 2]
    const baseCandidate = parts.slice(0, -2).join("-")
    if (baseCandidate && algo) {
      return <span className="text-sm text-muted-foreground">{baseCandidate}</span>
    }
  }
  return <span className="text-sm text-muted-foreground">--</span>
}

const columns: ColumnDef<ModelRecord, unknown>[] = [
  {
    accessorKey: "name",
    header: "Name",
    cell: ({ row }) => <ModelName model={row.original} />,
  },
  {
    id: "topic",
    header: "Topic",
    cell: ({ row }) => <TopicCell model={row.original} />,
  },
  {
    id: "base_model",
    header: "Base Model",
    cell: ({ row }) => <BaseModelCell model={row.original} />,
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
