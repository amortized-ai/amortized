import { useState } from "react"
import { useParams, useNavigate } from "react-router"
import { useModel, useDeleteModel } from "../api/use-models"
import { ModelDetail } from "./model-detail"
import { TableSkeleton } from "@/components/table-skeleton"
import { Button } from "@/components/ui/button"
import { ArrowLeft } from "lucide-react"
import { DeleteEntityDialog } from "@/components/delete-entity-dialog"

export function ModelDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const name = id ? decodeURIComponent(id) : null
  const { data: versions, isLoading } = useModel(name)
  const deleteMutation = useDeleteModel()
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)

  if (isLoading) {
    return (
      <div className="p-4">
        <TableSkeleton columns={4} />
      </div>
    )
  }

  if (!versions || versions.length === 0) {
    return (
      <div className="space-y-4">
        <Button variant="ghost" size="sm" onClick={() => void navigate("/models")}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back to Models
        </Button>
        <p className="text-sm text-muted-foreground">Model not found.</p>
      </div>
    )
  }

  return (
    <>
      <ModelDetail
        name={name!}
        versions={versions}
        onBack={() => void navigate("/models")}
        onDelete={() => setDeleteDialogOpen(true)}
      />
      <DeleteEntityDialog
        open={deleteDialogOpen}
        entityType="model"
        entityName={name!}
        onConfirm={() => {
          deleteMutation.mutate(name!, {
            onSuccess: () => {
              setDeleteDialogOpen(false)
              void navigate("/models")
            },
          })
        }}
        onCancel={() => setDeleteDialogOpen(false)}
        isPending={deleteMutation.isPending}
      />
    </>
  )
}
