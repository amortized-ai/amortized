import { useRef, useState } from "react"
import { Database, FileText, Paperclip } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { UploadDocumentDialog } from "@/features/documents/components/upload-document-dialog"
import { useAttachUpload } from "../hooks/use-attach-upload"

interface ChatAttachProps {
  // Posts a reference to the uploaded artifact into the conversation (usually sendMessage).
  onReference: (text: string) => void
  disabled?: boolean
}

export function ChatAttach({ onReference, disabled = false }: ChatAttachProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [docOpen, setDocOpen] = useState(false)
  const { attachDataset, attachDocument, isUploading } = useAttachUpload(onReference)

  function handleDatasetFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (file) attachDataset(file)
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            disabled={disabled || isUploading}
            aria-label="Attach data"
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
        onUploaded={(jobId, filename) => attachDocument(jobId, filename)}
      />
    </>
  )
}
