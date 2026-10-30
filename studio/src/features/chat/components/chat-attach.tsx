import { useRef, useState } from "react"
import { Database, FileText, Loader2, Paperclip } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { UploadDocumentDialog } from "@/features/documents/components/upload-document-dialog"
import type { AttachStatus } from "../hooks/use-attach-upload"

interface ChatAttachProps {
  onDataset: (file: File) => void
  onDocument: (jobId: string, filename: string) => void
  disabled?: boolean
}

export function ChatAttach({ onDataset, onDocument, disabled = false }: ChatAttachProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [docOpen, setDocOpen] = useState(false)

  function handleDatasetFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (file) onDataset(file)
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            disabled={disabled}
            aria-label="Attach data"
            title="Attach data"
            className="h-11 w-11 rounded-xl shrink-0 text-muted-foreground hover:text-foreground"
          >
            <Paperclip className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" side="top">
          <DropdownMenuItem onClick={() => fileInputRef.current?.click()}>
            <Database className="h-4 w-4" />
            Dataset
          </DropdownMenuItem>
          <DropdownMenuItem onClick={() => setDocOpen(true)}>
            <FileText className="h-4 w-4" />
            Document
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <input
        ref={fileInputRef}
        type="file"
        accept=".jsonl,.parquet"
        className="hidden"
        onChange={handleDatasetFile}
        data-testid="chat-attach-dataset-input"
      />
      <UploadDocumentDialog
        open={docOpen}
        onOpenChange={setDocOpen}
        showTrigger={false}
        onUploaded={(jobId, filename) => onDocument(jobId, filename)}
      />
    </>
  )
}

// Inline, non-blocking upload progress shown above the composer (one row per in-flight
// attach). Replaces the loading/success toasts, which covered the composer's Send button.
export function ChatAttachStatus({ statuses }: { statuses: AttachStatus[] }) {
  if (statuses.length === 0) return null
  return (
    <div className="space-y-1" data-testid="chat-attach-status">
      {statuses.map((s) => (
        <div key={s.id} className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin shrink-0" />
          <span className="truncate">{s.message}…</span>
        </div>
      ))}
    </div>
  )
}
