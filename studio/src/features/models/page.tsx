import { useState, useMemo, useEffect } from "react"
import { useNavigate, useSearchParams } from "react-router"
import { useModels } from "./api/use-models"
import { ModelTable } from "./components/model-table"
import { ModelDetailPanel } from "./components/model-detail-panel"
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
import { Button } from "@/components/ui/button"
import { Box, ArrowRight, GraduationCap, Layers, Rocket } from "lucide-react"
import { Link } from "react-router"
import { SearchInput } from "@/components/search-input"
import type { ModelRecord } from "@/types/api"

export default function ModelsPage() {
  const { data: models = [], isLoading, isError, error, refetch } = useModels()
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState("")
  const [selectedModel, setSelectedModel] = useState<ModelRecord | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()

  useEffect(() => {
    const runId = searchParams.get("run")
    const name = searchParams.get("name")
    if (models.length > 0 && (runId || name)) {
      const found = runId
        ? models.find((m) => m.run_id === runId)
        : models.find((m) => m.name === name)
      if (found) {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time URL param sync
        setSelectedModel(found)
        setDetailOpen(true)
        setSearchParams({}, { replace: true })
      }
    }
  }, [searchParams, models, setSearchParams])

  const filteredModels = useMemo(() => {
    if (!search.trim()) return models
    const q = search.toLowerCase()
    return models.filter((m) => m.name.toLowerCase().includes(q))
  }, [models, search])

  function handleSelectModel(model: ModelRecord) {
    setSelectedModel(model)
    setDetailOpen(true)
  }

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  return (
    <div className="space-y-6">
      <div data-tutorial="models-header" className="space-y-6">
        <PageHeader title="Models" description="Manage your fine-tuned models and track versions" />

        <div className="animate-message-in rounded-xl border bg-card p-4">
          <div className="grid gap-3 md:grid-cols-3 mb-3">
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <GraduationCap className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">From training</p>
                <p className="text-xs text-muted-foreground">Created by fine-tuning a base model</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#fff4cc] text-[#b98412] dark:bg-[#54330b]/40 dark:text-[#dca614]">
                <Layers className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Versioned</p>
                <p className="text-xs text-muted-foreground">Track iterations in MLflow registry</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#daf2f2] text-[#147878] dark:bg-[#003333]/40 dark:text-[#37a3a3]">
                <Rocket className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Deployable</p>
                <p className="text-xs text-muted-foreground">Ready to serve via vLLM or TGI</p>
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground border-t pt-3">
            Click any model to view versions and metadata.{" "}
            <Link to="/chat" className="inline-flex items-center gap-0.5 underline underline-offset-2 hover:text-foreground transition-colors">
              Build your first model <ArrowRight className="h-2.5 w-2.5" />
            </Link>
          </p>
        </div>
      </div>

      <SearchInput value={search} onChange={(v) => { setSearch(v); setPage(0) }} placeholder="Search models..." />

      {isLoading ? (
        <TableSkeleton columns={5} />
      ) : filteredModels.length > 0 ? (
        <ModelTable
          models={filteredModels}
          page={page}
          onPageChange={setPage}
          onSelectModel={handleSelectModel}
        />
      ) : (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Box /></EmptyMedia>
            <EmptyTitle>No models yet</EmptyTitle>
            <EmptyDescription>Trained models will appear here after running a training recipe.</EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <Button size="sm" onClick={() => navigate("/recipes")}>Create a Recipe</Button>
          </EmptyContent>
        </Empty>
      )}

      <ModelDetailPanel
        model={selectedModel}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </div>
  )
}
