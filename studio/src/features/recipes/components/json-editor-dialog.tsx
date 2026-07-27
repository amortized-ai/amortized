import { lazy, Suspense } from "react"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Code } from "lucide-react"

const JsonEditorInner = lazy(() => import("./json-editor-inner"))

interface JsonEditorDialogProps {
  jsonValue: string
  jsonError: string | null
  schema: Record<string, unknown> | null
  onJsonChange: (value: string) => void
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function JsonEditorDialog({
  jsonValue,
  jsonError,
  schema,
  onJsonChange,
  open,
  onOpenChange,
}: JsonEditorDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" data-testid="open-json-editor">
          <Code className="mr-2 h-4 w-4" />
          JSON Editor
        </Button>
      </DialogTrigger>
      <DialogContent className="!max-w-[50vw] !w-[50vw] !h-[42vh] flex flex-col overflow-hidden">
        <DialogHeader>
          <DialogTitle>Recipe Configuration</DialogTitle>
        </DialogHeader>
        <Suspense
          fallback={
            <div className="flex items-center justify-center flex-1 min-h-[300px] text-muted-foreground">
              Loading editor...
            </div>
          }
        >
          <JsonEditorInner
            jsonValue={jsonValue}
            jsonError={jsonError}
            schema={schema}
            onJsonChange={onJsonChange}
          />
        </Suspense>
      </DialogContent>
    </Dialog>
  )
}
