import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Upload, Loader2, CheckCircle2 } from "lucide-react"
import { toast } from "sonner"
import { useUploadDocument, useConvertDocumentUrl } from "../api/use-documents"
import { useJob } from "@/features/jobs"
import { useQueryClient } from "@tanstack/react-query"

const ACCEPTED_FORMATS = ".pdf,.docx,.pptx,.html,.txt,.md,.xlsx"
const OUTPUT_FORMATS = [
  { value: "md", label: "Markdown" },
  { value: "text", label: "Plain Text" },
  { value: "json", label: "JSON" },
  { value: "html", label: "HTML" },
] as const

const CHUNKER_TYPES = [
  { value: "sentence", label: "Sentence" },
  { value: "token", label: "Token" },
  { value: "recursive", label: "Recursive" },
] as const

const STATUS_LABELS: Record<string, string> = {
  queued: "Queued...",
  provisioning: "Provisioning...",
  running: "Processing document...",
}

export function UploadDocumentDialog() {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [url, setUrl] = useState("")
  const [outputFormat, setOutputFormat] = useState("md")
  const [chunkerType, setChunkerType] = useState("sentence")
  const [chunkSize, setChunkSize] = useState(2048)
  const [chunkOverlap, setChunkOverlap] = useState(200)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobFilename, setJobFilename] = useState("")
  const [toastId, setToastId] = useState<string | number | null>(null)
  const [handled, setHandled] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const queryClient = useQueryClient()

  const uploadMutation = useUploadDocument()
  const urlMutation = useConvertDocumentUrl()
  const { data: job } = useJob(jobId)

  const isUploading = uploadMutation.isPending || urlMutation.isPending
  const jobStatus = job?.status
  const isProcessing = jobId != null && jobStatus !== "succeeded" && jobStatus !== "failed"
  const isSucceeded = jobStatus === "succeeded"
  const isFailed = jobStatus === "failed"

  if (isSucceeded && !handled) {
    setHandled(true)
    void queryClient.invalidateQueries({ queryKey: ["documents"] })
    if (toastId) {
      toast.success(`${jobFilename || "Document"} processed`, { id: toastId })
      setToastId(null)
    }
    setTimeout(() => setOpen(false), 2000)
  }
  if (isFailed && !handled) {
    setHandled(true)
    if (toastId) {
      toast.error(`${jobFilename || "Document"} failed`, { id: toastId })
      setToastId(null)
    }
    setError(job?.error ?? "Document processing failed")
    setJobId(null)
  }

  function reset() {
    setFile(null)
    setUrl("")
    setOutputFormat("md")
    setChunkerType("sentence")
    setChunkSize(2048)
    setChunkOverlap(200)
    setJobId(null)
    setJobFilename("")
    setToastId(null)
    setHandled(false)
    setError(null)
    uploadMutation.reset()
    urlMutation.reset()
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  function handleOpenChange(v: boolean) {
    if (!v && jobId && !handled) {
      toast.loading(`Uploading ${jobFilename || "document"}...`, {
        duration: Infinity,
        id: "doc-upload",
      })
      setToastId("doc-upload")
    }
    setOpen(v)
    if (!v && !jobId) reset()
  }

  const chunkOptions = {
    output_format: outputFormat,
    chunker_type: chunkerType,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
  }

  function handleFileSubmit() {
    if (!file) return
    setError(null)
    setJobId(null)
    setJobFilename(file.name)
    uploadMutation.mutate(
      { file, options: chunkOptions },
      {
        onSuccess: (data) => setJobId(data.job_id),
        onError: (err) => {
          setError(err instanceof Error ? err.message : "Upload failed")
        },
      },
    )
  }

  function handleUrlSubmit() {
    if (!url.trim()) return
    setError(null)
    setJobId(null)
    setJobFilename(url.trim().split("/").pop() || "document")
    urlMutation.mutate(
      { url: url.trim(), options: chunkOptions },
      {
        onSuccess: (data) => setJobId(data.job_id),
        onError: (err) => {
          setError(err instanceof Error ? err.message : "Conversion failed")
        },
      },
    )
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button size="sm" data-testid="upload-document">
          <Upload className="h-4 w-4 mr-1" />
          Upload Document
        </Button>
      </DialogTrigger>
      <DialogContent onInteractOutside={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>Upload Document</DialogTitle>
          <DialogDescription>
            Upload a file or provide a URL to convert a document into a
            structured format for use in training data.
          </DialogDescription>
        </DialogHeader>

        {isUploading ? (
          <div className="flex flex-col items-center justify-center gap-3 py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Uploading file...</p>
          </div>
        ) : isProcessing ? (
          <div className="flex flex-col items-center justify-center gap-3 py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              {STATUS_LABELS[job?.status ?? ""] ?? "Processing..."}
            </p>
            <p className="text-xs text-muted-foreground">{jobFilename}</p>
          </div>
        ) : isSucceeded ? (
          <div className="flex flex-col items-center gap-3 py-6">
            <CheckCircle2 className="h-6 w-6 text-[#1e4f18] dark:text-[#5ba352]" />
            <div className="text-center space-y-1">
              <p className="text-sm font-medium">Document processed</p>
              <p className="text-xs text-muted-foreground">
                Ready for use in training data generation.
              </p>
            </div>
          </div>
        ) : (
          <Tabs defaultValue="upload" className="w-full">
            <TabsList className="w-full">
              <TabsTrigger value="upload" className="flex-1">
                Upload File
              </TabsTrigger>
              <TabsTrigger value="url" className="flex-1">
                From URL
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload">
              <div className="space-y-4 py-2">
                <div className="space-y-2">
                  <Label htmlFor="doc-file">File</Label>
                  <Input
                    id="doc-file"
                    ref={fileInputRef}
                    type="file"
                    accept={ACCEPTED_FORMATS}
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                    data-testid="doc-file-input"
                  />
                  <p className="text-xs text-muted-foreground">
                    PDF, DOCX, PPTX, HTML, TXT, Markdown, XLSX
                  </p>
                </div>
                <OutputFormatSelect
                  value={outputFormat}
                  onChange={setOutputFormat}
                />
                <ChunkSettings
                  chunkerType={chunkerType}
                  chunkSize={chunkSize}
                  chunkOverlap={chunkOverlap}
                  onChunkerTypeChange={setChunkerType}
                  onChunkSizeChange={setChunkSize}
                  onChunkOverlapChange={setChunkOverlap}
                />
              </div>
              <DialogFooter className="mt-4">
                <Button
                  variant="outline"
                  onClick={() => handleOpenChange(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleFileSubmit}
                  disabled={!file}
                  data-testid="doc-upload-submit"
                >
                  Submit
                </Button>
              </DialogFooter>
            </TabsContent>

            <TabsContent value="url">
              <div className="space-y-4 py-2">
                <div className="space-y-2">
                  <Label htmlFor="doc-url">URL</Label>
                  <Input
                    id="doc-url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://example.com/document.pdf"
                    data-testid="doc-url-input"
                  />
                </div>
                <OutputFormatSelect
                  value={outputFormat}
                  onChange={setOutputFormat}
                />
                <ChunkSettings
                  chunkerType={chunkerType}
                  chunkSize={chunkSize}
                  chunkOverlap={chunkOverlap}
                  onChunkerTypeChange={setChunkerType}
                  onChunkSizeChange={setChunkSize}
                  onChunkOverlapChange={setChunkOverlap}
                />
              </div>
              <DialogFooter className="mt-4">
                <Button
                  variant="outline"
                  onClick={() => handleOpenChange(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleUrlSubmit}
                  disabled={!url.trim()}
                  data-testid="doc-url-submit"
                >
                  Submit
                </Button>
              </DialogFooter>
            </TabsContent>
          </Tabs>
        )}

        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
      </DialogContent>
    </Dialog>
  )
}

function OutputFormatSelect({
  value,
  onChange,
}: {
  value: string
  onChange: (v: string) => void
}) {
  return (
    <div className="space-y-2">
      <Label>Output Format</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger data-testid="doc-format-select">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {OUTPUT_FORMATS.map((f) => (
            <SelectItem key={f.value} value={f.value}>
              {f.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

function ChunkSettings({
  chunkerType,
  chunkSize,
  chunkOverlap,
  onChunkerTypeChange,
  onChunkSizeChange,
  onChunkOverlapChange,
}: {
  chunkerType: string
  chunkSize: number
  chunkOverlap: number
  onChunkerTypeChange: (v: string) => void
  onChunkSizeChange: (v: number) => void
  onChunkOverlapChange: (v: number) => void
}) {
  const supportsOverlap = chunkerType === "token" || chunkerType === "sentence"
  return (
    <div className="space-y-3">
      <div className="space-y-2">
        <Label>Chunker</Label>
        <Select value={chunkerType} onValueChange={onChunkerTypeChange}>
          <SelectTrigger data-testid="doc-chunker-type">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CHUNKER_TYPES.map((c) => (
              <SelectItem key={c.value} value={c.value}>
                {c.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div className="space-y-2">
          <Label htmlFor="chunk-size">Chunk Size</Label>
          <Input
            id="chunk-size"
            type="number"
            min={64}
            max={8192}
            value={chunkSize}
            onChange={(e) => onChunkSizeChange(Number(e.target.value) || 2048)}
            data-testid="doc-chunk-size"
          />
        </div>
        <div className="space-y-2">
          <Label htmlFor="chunk-overlap">Overlap</Label>
          <Input
            id="chunk-overlap"
            type="number"
            min={0}
            value={chunkOverlap}
            disabled={!supportsOverlap}
            onChange={(e) => onChunkOverlapChange(Number(e.target.value) || 0)}
            data-testid="doc-chunk-overlap"
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Overlap applies to token and sentence chunkers only.
      </p>
    </div>
  )
}
