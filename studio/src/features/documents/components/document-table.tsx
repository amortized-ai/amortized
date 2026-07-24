import { type ColumnDef } from "@tanstack/react-table"
import { FileText, Eye } from "lucide-react"
import { useMemo } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { EmptyState } from "@/components/empty-state"
import { DataTable } from "@/components/data-table"
import type { DocumentRecord } from "@/types/api"

interface DocumentTableProps {
  documents: DocumentRecord[]
  page: number
  onPageChange: (page: number) => void
  onSelectDocument: (document: DocumentRecord) => void
}

export function DocumentTable({
  documents,
  page,
  onPageChange,
  onSelectDocument,
}: DocumentTableProps) {
  const sorted = useMemo(
    () =>
      [...documents].sort((a, b) => {
        if (!a.created_at || !b.created_at) return 0
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      }),
    [documents],
  )

  const columns: ColumnDef<DocumentRecord, unknown>[] = useMemo(
    () => [
      {
        accessorKey: "filename",
        header: "Filename",
        cell: ({ row }) => (
          <span className="font-medium">{row.original.filename}</span>
        ),
      },
      {
        accessorKey: "format",
        header: "Format",
        cell: ({ row }) => (
          <Badge variant="secondary" className="uppercase text-[10px]">
            {row.original.format}
          </Badge>
        ),
      },
      {
        accessorFn: (row) =>
          row.created_at
            ? new Date(row.created_at).toLocaleString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
                hour: "numeric",
                minute: "2-digit",
              })
            : "--",
        id: "created_at",
        header: "Created",
        cell: ({ getValue }) => (
          <span className="text-sm text-muted-foreground whitespace-nowrap">
            {getValue() as string}
          </span>
        ),
      },
      {
        id: "actions",
        header: "",
        cell: ({ row }) => (
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation()
              onSelectDocument(row.original)
            }}
            aria-label={`View ${row.original.filename}`}
          >
            <Eye className="h-4 w-4" />
          </Button>
        ),
        enableSorting: false,
      },
    ],
    [onSelectDocument],
  )

  const paginationState = useMemo(
    () => ({ pageIndex: page, pageSize: 10 }),
    [page],
  )

  return (
    <DataTable
      columns={columns}
      data={sorted}
      onRowClick={onSelectDocument}
      pagination={paginationState}
      onPaginationChange={(updater) => {
        const next = typeof updater === "function" ? updater(paginationState) : updater
        onPageChange(next.pageIndex)
      }}
      rowTestId={(doc) => `document-row-${doc.document_id}`}
      emptyState={
        <EmptyState
          icon={FileText}
          title="No documents yet"
          description="Upload a document to get started."
        />
      }
    />
  )
}
