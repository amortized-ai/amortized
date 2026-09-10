import { useState, useEffect } from "react"
import { useSearchParams } from "react-router"
import { SearchInput } from "@/components/search-input"
import { ErrorState } from "@/components/error-state"
import { PageHeader } from "@/components/page-header"
import { TableSkeleton } from "@/components/table-skeleton"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
} from "@/components/ui/empty"
import { ClipboardCheck, MessageSquare } from "lucide-react"
import { Link } from "react-router"
import { useEvaluations } from "./api/use-evaluations"
import { EvaluationTable } from "./components/evaluation-table"
import { EvaluationDetailPanel } from "./components/evaluation-detail-panel"
import type { EvaluationGroup } from "@/lib/api-client"

export default function EvaluationPage() {
  const { data: groups = [], isLoading, isError, error, refetch } = useEvaluations()
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState("")
  const [selectedGroup, setSelectedGroup] = useState<EvaluationGroup | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const [searchParams, setSearchParams] = useSearchParams()

  useEffect(() => {
    const groupId = searchParams.get("group")
    if (groups.length > 0 && groupId) {
      const found = groups.find((g) => g.id === groupId)
      if (found) {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time URL param sync
        setSelectedGroup(found)
        setDetailOpen(true)
        setSearchParams({}, { replace: true })
      }
    }
  }, [searchParams, groups, setSearchParams])

  const filtered = search.trim()
    ? groups.filter(
        (g) =>
          g.dataset.name.toLowerCase().includes(search.toLowerCase()) ||
          g.evals.some((e) => e.model.toLowerCase().includes(search.toLowerCase())),
      )
    : groups

  function handleSelectGroup(group: EvaluationGroup) {
    setSelectedGroup(group)
    setDetailOpen(true)
  }

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  return (
    <div className="space-y-6">
      <div className="space-y-6">
        <PageHeader
          title="Evaluation"
          description="Compare model performance on the same eval datasets and metrics"
        />

        <div className="animate-message-in rounded-xl border bg-card p-4">
          <div className="grid gap-3 md:grid-cols-3 mb-3">
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e6f7f2] text-[#0d7a5f] dark:bg-[#003326]/40 dark:text-[#3fbf9a]">
                <ClipboardCheck className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Consistent metrics</p>
                <p className="text-xs text-muted-foreground">
                  Every eval on a dataset reuses its metric set — models stay comparable
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <ClipboardCheck className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Model comparison</p>
                <p className="text-xs text-muted-foreground">
                  Base and trained models side by side, per metric
                </p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
                <MessageSquare className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Run via Morty</p>
                <p className="text-xs text-muted-foreground">
                  Ask Morty to evaluate a model — serving is handled automatically
                </p>
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground border-t pt-3">
            Click a row to compare models.{" "}
            <Link to="/chat" className="inline-flex items-center gap-0.5 underline underline-offset-2 hover:text-foreground transition-colors">
              Start an evaluation <ClipboardCheck className="h-2.5 w-2.5" />
            </Link>
          </p>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex-1">
          <SearchInput
            value={search}
            onChange={(v) => {
              setSearch(v)
              setPage(0)
            }}
            placeholder="Search by dataset or model..."
          />
        </div>
      </div>

      {isLoading ? (
        <TableSkeleton columns={5} />
      ) : filtered.length > 0 ? (
        <EvaluationTable
          groups={filtered}
          page={page}
          onPageChange={setPage}
          onSelectGroup={handleSelectGroup}
        />
      ) : (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <ClipboardCheck />
            </EmptyMedia>
            <EmptyTitle>No evaluations yet</EmptyTitle>
            <EmptyDescription>
              Ask Morty to evaluate a model — comparisons will appear here.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      )}

      <EvaluationDetailPanel
        group={selectedGroup}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </div>
  )
}
