import { useState, useCallback } from "react"
import { useNavigate } from "react-router"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"
import { JsonTreeViewer } from "@/components/json-tree-viewer"
import { RecipeConfigForm } from "./recipe-config-form"
import { stripPlaceholders } from "../lib/placeholders"
import { collectMissingRequired } from "../lib/recipe-validation"
import { useCreateJobFromConfig, useConfigSchemas } from "../api/use-recipes"
import type { JsonSchema } from "../api/use-recipes"
import {
  formatRecipeType,
  recipeTypeClassName,
} from "../lib/format"
import { ArrowLeft, Play, Code2, FormInput } from "lucide-react"
import type { Job, JobType } from "@/types/api"

interface RecipeDetailPanelProps {
  job: Job
  recipeName: string
  onBack: () => void
  // Per-recipe required top-level fields (from a starter template's _meta.required).
  requiredKeys?: string[]
}

export function RecipeDetailPanel({ job, recipeName, onBack, requiredKeys }: RecipeDetailPanelProps) {
  const navigate = useNavigate()
  const createMutation = useCreateJobFromConfig()
  const { data: schemas } = useConfigSchemas()
  // Strip placeholder tokens (agent-resolution hints, e.g. "<from-list_models>") on
  // load so they are never a value and never submitted; pickers show an empty state.
  const [config, setConfig] = useState<Record<string, unknown>>(
    () => stripPlaceholders(structuredClone(job.config ?? {})),
  )
  const [executeDialogOpen, setExecuteDialogOpen] = useState(false)
  const [submittedJobId, setSubmittedJobId] = useState<string | null>(null)
  const [activeView, setActiveView] = useState<"form" | "json">("form")

  // Block execution while any required field is empty: schema-required (the schema
  // is the source of truth) plus any per-recipe required fields (requiredKeys).
  const typeSchema = schemas ? (schemas[job.type] as JsonSchema | undefined) : null
  const missingRequired = collectMissingRequired(config, typeSchema, requiredKeys)

  const handleExecute = useCallback(() => {
    createMutation.mutate(
      { type: job.type as JobType, config },
      {
        onSuccess: (newJob) => {
          setSubmittedJobId(newJob.id)
        },
        onError: (err) => {
          toast.error(`Failed to create job: ${err.message}`)
          setExecuteDialogOpen(false)
        },
      },
    )
  }, [config, job.type, createMutation])

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b px-2 py-3 shrink-0">
        <div className="flex items-center gap-3 min-w-0">
          <Button variant="ghost" size="sm" onClick={onBack} className="shrink-0">
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="text-lg font-semibold truncate">{recipeName}</h1>
              <Badge className={recipeTypeClassName(job.type)}>
                {formatRecipeType(job.type)}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              From job {job.id.slice(0, 8)}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center rounded-lg border p-0.5">
            <Button
              variant={activeView === "form" ? "secondary" : "ghost"}
              size="sm"
              className="h-7 px-2.5 text-xs"
              onClick={() => setActiveView("form")}
            >
              <FormInput className="mr-1.5 h-3.5 w-3.5" />
              Builder
            </Button>
            <Button
              variant={activeView === "json" ? "secondary" : "ghost"}
              size="sm"
              className="h-7 px-2.5 text-xs"
              onClick={() => setActiveView("json")}
            >
              <Code2 className="mr-1.5 h-3.5 w-3.5" />
              JSON
            </Button>
          </div>

          <Button
            size="sm"
            onClick={() => setExecuteDialogOpen(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Play className="mr-1.5 h-3.5 w-3.5" />
            Execute
          </Button>
        </div>
      </div>

      {/* Split panel */}
      <div className="flex-1 min-h-0">
        {activeView === "json" ? (
          <ScrollArea className="h-full">
            <div className="p-4">
              <div className="rounded-lg border bg-muted/30 p-4">
                <JsonTreeViewer data={config} collapsed={3} />
              </div>
            </div>
          </ScrollArea>
        ) : (
          <ResizablePanelGroup orientation="horizontal">
            <ResizablePanel defaultSize={40} minSize={25}>
              <ScrollArea className="h-full">
                <div className="p-4">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
                    Config Preview
                  </p>
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <JsonTreeViewer data={config} collapsed={2} />
                  </div>
                </div>
              </ScrollArea>
            </ResizablePanel>

            <ResizableHandle withHandle />

            <ResizablePanel defaultSize={60} minSize={35}>
              <ScrollArea className="h-full">
                <div className="p-4">
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">
                    Configuration
                  </p>
                  <RecipeConfigForm
                    type={job.type as JobType}
                    config={config}
                    onChange={setConfig}
                    schema={schemas}
                    requiredKeys={requiredKeys}
                  />
                </div>
              </ScrollArea>
            </ResizablePanel>
          </ResizablePanelGroup>
        )}
      </div>

      {/* Execute confirmation dialog */}
      <Dialog
        open={executeDialogOpen}
        onOpenChange={(open) => {
          setExecuteDialogOpen(open)
          if (!open) setSubmittedJobId(null)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {submittedJobId ? "Job Submitted" : "Execute Recipe"}
            </DialogTitle>
            <DialogDescription>
              {submittedJobId
                ? `Job ${submittedJobId.slice(0, 8)} has been created.`
                : `This will create a new ${formatRecipeType(job.type)} job with the current configuration.`}
            </DialogDescription>
          </DialogHeader>

          {!submittedJobId && missingRequired.length > 0 && (
            <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm">
              <p className="font-medium text-destructive">Fill required fields before running:</p>
              <ul className="mt-1 list-disc pl-5 text-destructive/90">
                {missingRequired.map((field) => (
                  <li key={field}>{field}</li>
                ))}
              </ul>
            </div>
          )}

          {!submittedJobId && (
            <div className="max-h-[300px] overflow-auto rounded-lg border bg-muted/30 p-3">
              <JsonTreeViewer data={config} collapsed={2} />
            </div>
          )}

          <DialogFooter>
            {submittedJobId ? (
              <Button
                onClick={() => {
                  setExecuteDialogOpen(false)
                  setTimeout(() => {
                    setSubmittedJobId(null)
                    navigate(
                      `/jobs?job=${encodeURIComponent(submittedJobId)}`,
                    )
                  }, 200)
                }}
              >
                View Job
              </Button>
            ) : (
              <>
                <Button
                  variant="outline"
                  onClick={() => setExecuteDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button
                  onClick={handleExecute}
                  disabled={createMutation.isPending || missingRequired.length > 0}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {createMutation.isPending ? "Submitting..." : "Submit Job"}
                </Button>
              </>
            )}
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
