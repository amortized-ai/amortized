import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog"
import { X, FileText, Copy, Hash, Calendar, Loader2, Trash2 } from "lucide-react"
import { useState } from "react"
import { useQueryClient } from "@tanstack/react-query"
import { useDocumentContent, useDocumentChunks, useDeleteDocument } from "../api/use-documents"
import { EditableTitle } from "@/components/editable-title"
import { DeleteEntityDialog } from "@/components/delete-entity-dialog"
import { setMlflowRunTag } from "@/lib/api-client"
import type { DocumentRecord } from "@/types/api"

interface DocumentDetailPanelProps {
  document: DocumentRecord | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function DocumentDetailPanel({
  document,
  open,
  onOpenChange,
}: DocumentDetailPanelProps) {
  const queryClient = useQueryClient()
  const deleteMutation = useDeleteDocument()
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false)
  if (!document) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        variant="panel"
        showCloseButton={false}
        className="!max-w-[900px] !w-[900px] !h-[80vh] flex flex-col !gap-0 p-0 overflow-hidden rounded-xl"
      >
        <PanelHeader
          document={document}
          onClose={() => onOpenChange(false)}
          onDelete={() => setDeleteDialogOpen(true)}
          onRename={async (newName) => {
            const runId = document.mlflow_run_id || document.document_id
            await setMlflowRunTag(runId, "filename", newName)
            queryClient.invalidateQueries({ queryKey: ["documents"] })
          }}
        />
        <PanelBody document={document} />

        <DeleteEntityDialog
          open={deleteDialogOpen}
          entityType="document"
          entityName={document.filename}
          onConfirm={() => {
            deleteMutation.mutate(document.document_id, {
              onSuccess: () => {
                setDeleteDialogOpen(false)
                onOpenChange(false)
              },
            })
          }}
          onCancel={() => setDeleteDialogOpen(false)}
          isPending={deleteMutation.isPending}
        />
      </DialogContent>
    </Dialog>
  )
}

function PanelHeader({
  document,
  onClose,
  onDelete,
  onRename,
}: {
  document: DocumentRecord
  onClose: () => void
  onDelete: () => void
  onRename: (newName: string) => Promise<void>
}) {
  return (
    <div className="flex items-center justify-between gap-4 px-6 py-4 border-b shrink-0">
      <div className="flex items-center gap-3 min-w-0 flex-1 overflow-hidden">
        <DialogTitle asChild>
          <EditableTitle
            value={document.filename}
            className="text-base font-semibold truncate"
            onSave={onRename}
          />
        </DialogTitle>
        <Badge
          variant="secondary"
          className="shrink-0 gap-1 bg-[#e0f0ff] text-[#0066cc] border-0 dark:bg-[#003366]/40 dark:text-[#4394e5]"
        >
          <FileText className="h-3 w-3" />
          {document.format.toUpperCase()}
        </Badge>
      </div>
      <div className="flex items-center gap-2 shrink-0">
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0 opacity-70 hover:opacity-100 hover:text-destructive"
          onClick={onDelete}
        >
          <Trash2 className="h-4 w-4" />
          <span className="sr-only">Delete</span>
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-8 w-8 p-0 opacity-70 hover:opacity-100"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </Button>
      </div>
    </div>
  )
}

function PanelBody({ document }: { document: DocumentRecord }) {
  const {
    data: content,
    isLoading,
    isError,
    error,
  } = useDocumentContent(document.document_id)
  const { data: chunksData } = useDocumentChunks(document.document_id)

  return (
    <Tabs defaultValue="content" className="flex flex-col flex-1 min-h-0">
      <div className="px-6 pt-3 shrink-0">
        <TabsList className="w-full">
          <TabsTrigger
            value="content"
            className="flex-1 transition-all duration-200"
          >
            Content
          </TabsTrigger>
          <TabsTrigger
            value="chunks"
            className="flex-1 transition-all duration-200"
          >
            Chunks{chunksData ? ` (${chunksData.chunks.length})` : ""}
          </TabsTrigger>
          <TabsTrigger
            value="info"
            className="flex-1 transition-all duration-200"
          >
            Info
          </TabsTrigger>
        </TabsList>
      </div>

      <TabsContent
        value="content"
        className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
      >
        <ContentTab content={content} isLoading={isLoading} isError={isError} error={error} />
      </TabsContent>

      <TabsContent
        value="chunks"
        className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
      >
        <ChunksTab chunksData={chunksData} />
      </TabsContent>

      <TabsContent
        value="info"
        className="mt-0 flex-1 min-h-0 overflow-y-auto px-6 py-4"
      >
        <InfoTab document={document} content={content} chunksData={chunksData} />
      </TabsContent>
    </Tabs>
  )
}

function ContentTab({
  content,
  isLoading,
  isError,
  error,
}: {
  content: import("@/types/api").DocumentUploadResponse | undefined
  isLoading: boolean
  isError: boolean
  error: Error | null
}) {
  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground gap-2">
        <Loader2 className="h-4 w-4 animate-spin" />
        Loading content...
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
        <p>Could not load document content.</p>
        <p className="text-xs font-mono">{(error as Error)?.message}</p>
      </div>
    )
  }

  if (!content?.content) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
        No content available.
      </div>
    )
  }

  if (content.format === "md") {
    return (
      <pre className="whitespace-pre-wrap break-words text-sm leading-relaxed font-mono">
        {content.content}
      </pre>
    )
  }

  return (
    <pre className="whitespace-pre-wrap break-words text-sm leading-relaxed font-mono bg-muted/30 rounded-lg p-4">
      {content.content}
    </pre>
  )
}

function InfoTab({
  document,
  content,
  chunksData,
}: {
  document: DocumentRecord
  content: import("@/types/api").DocumentUploadResponse | undefined
  chunksData: import("@/types/api").DocumentChunksResponse | undefined
}) {
  const [copiedField, setCopiedField] = useState<string | null>(null)

  function handleCopy(field: string, value: string) {
    void navigator.clipboard.writeText(value)
    setCopiedField(field)
    setTimeout(() => setCopiedField(null), 2000)
  }

  return (
    <div className="space-y-0">
      <MetadataRow
        label="Document ID"
        value={
          <span className="flex items-center gap-1.5">
            <Hash className="h-3 w-3 text-muted-foreground shrink-0" />
            <span className="font-mono text-xs">{document.document_id}</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-5 w-5 p-0 shrink-0"
              onClick={() => handleCopy("document_id", document.document_id)}
              aria-label="Copy document ID"
            >
              <Copy className="h-3 w-3" />
            </Button>
            {copiedField === "document_id" && (
              <span className="text-xs text-muted-foreground">Copied!</span>
            )}
          </span>
        }
      />
      {document.mlflow_run_id && (
        <MetadataRow
          label="MLflow Run ID"
          value={
            <span className="flex items-center gap-1.5">
              <Hash className="h-3 w-3 text-muted-foreground shrink-0" />
              <span className="font-mono text-xs">
                {document.mlflow_run_id}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-5 w-5 p-0 shrink-0"
                onClick={() =>
                  handleCopy("mlflow_run_id", document.mlflow_run_id!)
                }
                aria-label="Copy MLflow run ID"
              >
                <Copy className="h-3 w-3" />
              </Button>
              {copiedField === "mlflow_run_id" && (
                <span className="text-xs text-muted-foreground">Copied!</span>
              )}
            </span>
          }
        />
      )}
      <MetadataRow label="Filename" value={document.filename} />
      <MetadataRow
        label="Format"
        value={
          <Badge variant="outline" className="text-xs">
            {document.format}
          </Badge>
        }
      />
      {document.created_at && (
        <MetadataRow
          label="Created"
          value={
            <span className="flex items-center gap-1.5">
              <Calendar className="h-3 w-3 text-muted-foreground shrink-0" />
              {new Date(document.created_at).toLocaleString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
                hour: "numeric",
                minute: "2-digit",
              })}
            </span>
          }
        />
      )}
      {chunksData && chunksData.chunks.length > 0 && (() => {
        const totalTokens = chunksData.chunks.reduce(
          (sum, c) => sum + (c.num_tokens ?? 0), 0
        )
        const maxTokens = Math.max(
          ...chunksData.chunks.map((c) => c.num_tokens ?? 0)
        )
        return (
          <>
            <MetadataRow
              label="Total Tokens"
              value={totalTokens.toLocaleString()}
            />
            <MetadataRow
              label="Chunks"
              value={chunksData.chunks.length}
            />
            <MetadataRow
              label="Max Tokens per Chunk"
              value={maxTokens.toLocaleString()}
            />
          </>
        )
      })()}
      {content && (
        <>
          <MetadataRow
            label="Processing Time"
            value={`${content.processing_time.toFixed(1)}s`}
          />
          {content.warnings.length > 0 && (
            <MetadataRow
              label="Warnings"
              value={
                <span className="text-amber-600 dark:text-amber-400">
                  {content.warnings.join(", ")}
                </span>
              }
            />
          )}
        </>
      )}
    </div>
  )
}

function ChunksTab({
  chunksData,
}: {
  chunksData: import("@/types/api").DocumentChunksResponse | undefined
}) {
  if (!chunksData || chunksData.chunks.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm text-muted-foreground">
        No chunks available. This document may have been uploaded before chunking was enabled.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {chunksData.chunks.map((chunk) => (
        <div
          key={chunk.chunk_index}
          className="rounded-lg border border-border/50 p-3 space-y-2"
        >
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span className="font-medium">
              Chunk {chunk.chunk_index}
              {chunk.headings.length > 0 && (
                <span className="ml-2 font-normal">{chunk.headings[0]}</span>
              )}
            </span>
            {chunk.num_tokens != null && (
              <span>{chunk.num_tokens} tokens</span>
            )}
          </div>
          <pre className="whitespace-pre-wrap break-words text-sm leading-relaxed font-mono bg-muted/30 rounded p-2">
            {chunk.text}
          </pre>
        </div>
      ))}
    </div>
  )
}

function MetadataRow({
  label,
  value,
}: {
  label: string
  value: React.ReactNode
}) {
  return (
    <div className="flex items-start justify-between gap-4 px-2 py-2.5 -mx-2 rounded-lg transition-colors duration-150 hover:bg-muted/30 border-b border-border/30 last:border-0">
      <span className="text-sm text-muted-foreground shrink-0">{label}</span>
      <span className="text-sm text-right">{value}</span>
    </div>
  )
}
