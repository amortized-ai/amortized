import { useState, useMemo, useEffect } from "react"
import { useSearchParams } from "react-router"
import { SearchInput } from "@/components/search-input"
import { useDocuments } from "./api/use-documents"
import { DocumentTable } from "./components/document-table"
import { DocumentDetailPanel } from "./components/document-detail-panel"
import { UploadDocumentDialog } from "./components/upload-document-dialog"
import { ErrorState } from "@/components/error-state"
import { PageHeader } from "@/components/page-header"
import { TableSkeleton } from "@/components/table-skeleton"

import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
} from "@/components/ui/empty"
import { FileText, ScanLine, ArrowRight } from "lucide-react"
import type { DocumentRecord } from "@/types/api"

export default function DocumentsPage() {
  const { data: documents = [], isLoading, isError, error, refetch } = useDocuments()
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState("")
  const [selectedDocument, setSelectedDocument] = useState<DocumentRecord | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const [searchParams, setSearchParams] = useSearchParams()

  useEffect(() => {
    const docParam = searchParams.get("doc")
    if (!docParam) return
    if (documents.length === 0) {
      void refetch()
      return
    }
    const match = documents.find(
      (d) => d.document_id === docParam || d.mlflow_run_id === docParam,
    )
    if (match) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time URL param sync
      setSelectedDocument(match)
      setDetailOpen(true)
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        next.delete("doc")
        return next
      }, { replace: true })
    } else {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        next.delete("doc")
        return next
      }, { replace: true })
    }
  }, [searchParams, documents, setSearchParams, refetch])

  const filteredDocuments = useMemo(() => {
    if (!search.trim()) return documents
    const q = search.toLowerCase()
    return documents.filter((d) => d.filename.toLowerCase().includes(q))
  }, [documents, search])

  function handleSelectDocument(doc: DocumentRecord) {
    setSelectedDocument(doc)
    setDetailOpen(true)
  }

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  return (
    <div className="space-y-6">
      <div className="space-y-4" data-tutorial="documents-header">
        <PageHeader
          title="Documents"
          description="Upload and manage documents for synthetic data generation."
          actions={<UploadDocumentDialog />}
        />

        <div className="animate-message-in rounded-xl border bg-card p-4">
          <div className="grid gap-3 md:grid-cols-3 mb-3">
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
                <FileText className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Parse PDFs & DOCX</p>
                <p className="text-xs text-muted-foreground">Extract structured content from uploaded documents</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <ScanLine className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">OCR & Tables</p>
                <p className="text-xs text-muted-foreground">Recognize text in scans and extract table data</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#fff4cc] text-[#b98412] dark:bg-[#54330b]/40 dark:text-[#dca614]">
                <ArrowRight className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">SDG Pipeline Input</p>
                <p className="text-xs text-muted-foreground">Use parsed content as grounding for data generation</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <SearchInput value={search} onChange={(v) => { setSearch(v); setPage(0) }} placeholder="Search documents..." />

      {isLoading ? (
        <TableSkeleton columns={4} />
      ) : filteredDocuments.length > 0 ? (
        <DocumentTable
          documents={filteredDocuments}
          page={page}
          onPageChange={setPage}
          onSelectDocument={handleSelectDocument}
        />
      ) : (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon"><FileText /></EmptyMedia>
            <EmptyTitle>No documents yet</EmptyTitle>
            <EmptyDescription>Upload a PDF or DOCX to get started.</EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <UploadDocumentDialog />
          </EmptyContent>
        </Empty>
      )}

      <DocumentDetailPanel
        document={selectedDocument}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />

    </div>
  )
}
