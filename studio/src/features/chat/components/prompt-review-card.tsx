import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { FileText, ChevronDown, ChevronRight } from "lucide-react"

interface PromptReviewData {
  title?: string
  prompt: string
  purpose?: string
}

interface PromptReviewCardProps {
  data: PromptReviewData
}

export function PromptReviewCard({ data }: PromptReviewCardProps) {
  const [open, setOpen] = useState(true)
  const title = data.title?.trim() || "System prompt"

  return (
    <Card className="border-rh-purple/30 dark:border-rh-purple-dark/50 bg-rh-purple-light/30 dark:bg-rh-purple-dark/10 py-0 gap-0">
      <CardContent className="p-4">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex w-full items-center justify-between gap-2"
        >
          <span className="flex items-center gap-2">
            <FileText className="h-4 w-4 text-rh-purple dark:text-rh-purple" />
            <span className="text-sm font-semibold text-foreground">{title}</span>
          </span>
          {open ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
        </button>

        {data.purpose?.trim() && (
          <p className="mt-1 text-xs text-muted-foreground">{data.purpose}</p>
        )}

        {open && (
          <pre className="mt-3 max-h-96 overflow-auto rounded-lg border border-rh-purple-light dark:border-rh-purple-dark/30 bg-background/60 dark:bg-rh-gray-90/30 p-3 text-xs leading-relaxed text-foreground whitespace-pre-wrap break-words">
            {data.prompt}
          </pre>
        )}
      </CardContent>
    </Card>
  )
}
