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
import { Upload, Loader2, CheckCircle2, XCircle, AlertCircle } from "lucide-react"
import { toast } from "sonner"
import { Link } from "react-router"
import { useUploadDocument, useConvertDocumentUrl } from "../api/use-documents"
import { useJob } from "@/features/jobs"
import { getJob } from "@/lib/api-client"
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

function extractErrorMessage(err: unknown): { message: string; hint?: string } {
  const raw = err instanceof Error ? err.message : String(err)

  if (raw.includes("503") || raw.includes("service unavailable"))
    return { message: "Document processing is not configured", hint: "The document processing service is not running. Check that the docling sidecar is enabled in your backend configuration." }
  if (raw.includes("502") || raw.includes("bad gateway"))
    return { message: "Could not reach the backend", hint: "The server returned a 502 error. This usually means the backend process crashed or is restarting." }
  if (raw.includes("413") || raw.includes("too large"))
    return { message: "File is too large", hint: "The file exceeds the maximum upload size. Try a smaller file or increase the server's upload limit." }
  if (raw.includes("415") || raw.includes("unsupported"))
    return { message: "Unsupported file format", hint: "This file type is not supported. Supported formats: PDF, DOCX, PPTX, HTML, TXT, Markdown, XLSX." }

  return { message: raw || "Upload failed" }
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
  const [minimized, setMinimized] = useState(false)
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
    setTimeout(() => setOpen(false), 1500)
  }
  if (isFailed && !handled) {
    setHandled(true)
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
    setMinimized(false)
    setHandled(false)
    setError(null)
    uploadMutation.reset()
    urlMutation.reset()
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  function handleOpenChange(v: boolean) {
    if (!v && jobId && !handled) {
      const name = jobFilename || "document"
      const trackingJobId = jobId
      toast.loading(`Uploading ${name}...`, { duration: Infinity, id: "doc-upload" })
      setMinimized(true)

      const interval = setInterval(async () => {
        try {
          const j = await getJob(trackingJobId)
          if (j.status === "succeeded") {
            clearInterval(interval)
            toast.success(`${name} processed`, { id: "doc-upload" })
            void queryClient.invalidateQueries({ queryKey: ["documents"] })
          } else if (j.status === "failed") {
            clearInterval(interval)
            toast.error(`${name} failed`, { id: "doc-upload" })
          }
        } catch {
          clearInterval(interval)
          toast.error(`${name} - lost connection`, { id: "doc-upload" })
        }
      }, 3000)
    }
    setOpen(v)
    if (!v && !minimized && !jobId) reset()
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
          const { message, hint } = extractErrorMessage(err)
          setError(hint ? `${message}. ${hint}` : message)
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
          const { message, hint } = extractErrorMessage(err)
          setError(hint ? `${message}. ${hint}` : message)
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
            {jobId && (
              <Link to={`/jobs?job=${jobId}`} className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground transition-colors">
                View job details
              </Link>
            )}
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
          <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3">
            <XCircle className="h-4 w-4 text-destructive shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0 space-y-1">
              <p className="text-sm text-destructive">{error}</p>
              {jobId && (
                <Link to={`/jobs?job=${jobId}`} className="inline-flex items-center gap-1 text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground transition-colors">
                  <AlertCircle className="h-3 w-3" />
                  View job details
                </Link>
              )}
            </div>
          </div>
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
