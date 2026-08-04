import { useMemo, useState, useEffect } from "react"
import { Link, useSearchParams } from "react-router"
import { useJobs, useJob } from "./api/use-jobs"
import { CostBreakdownChart } from "./components/cost-breakdown-chart"
import { useEntityNamesStore } from "@/stores/entity-names-store"
import { useJobFilters } from "./hooks/use-job-filters"
import { JobTable } from "./components/job-table"
import { JobDetailPanel } from "./components/job-detail-panel"
import { FilterChips, TYPE_COLORS, STATUS_COLORS } from "./components/filter-chips"
import { ErrorState } from "@/components/error-state"
import { PageHeader } from "@/components/page-header"
import { TableSkeleton } from "@/components/table-skeleton"
import { Button } from "@/components/ui/button"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
} from "@/components/ui/empty"
import { Briefcase, ArrowRight, GraduationCap, Sparkles } from "lucide-react"
import { SearchInput } from "@/components/search-input"
import type { Job, JobType, JobStatus } from "@/types/api"

const TYPE_OPTIONS = [
  { value: "sdg" as JobType, label: "SDG", color: TYPE_COLORS.sdg },
  { value: "training" as JobType, label: "Training", color: TYPE_COLORS.training },
]

const STATUS_OPTIONS = [
  { value: "queued" as JobStatus, label: "Queued", color: STATUS_COLORS.queued },
  { value: "provisioning" as JobStatus, label: "Provisioning", color: STATUS_COLORS.provisioning },
  { value: "running" as JobStatus, label: "Running", color: STATUS_COLORS.running },
  { value: "succeeded" as JobStatus, label: "Succeeded", color: STATUS_COLORS.succeeded },
  { value: "failed" as JobStatus, label: "Failed", color: STATUS_COLORS.failed },
  { value: "cancelled" as JobStatus, label: "Cancelled", color: STATUS_COLORS.cancelled },
]

export default function JobsPage() {
  const { typeFilter, statusFilter, setTypeFilter, setStatusFilter } =
    useJobFilters()

  const { data: jobs = [], isLoading, isError, error, refetch } = useJobs()
  const entityNames = useEntityNamesStore((s) => s.names)

  const [searchParams, setSearchParams] = useSearchParams()

  const [page, setPage] = useState(0)
  const [search, setSearch] = useState("")
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const { data: selectedJob = null } = useJob(selectedJobId)

  useEffect(() => {
    const jobParam = searchParams.get("job")
    if (jobParam) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time URL param sync
      setSelectedJobId(jobParam)
      setDetailOpen(true)
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        next.delete("job")
        return next
      }, { replace: true })
    }
  }, [searchParams, setSearchParams])

  const filteredJobs = useMemo(() => {
    let result = jobs
    if (typeFilter.length > 0) {
      result = result.filter((j) => typeFilter.includes(j.type))
    }
    if (statusFilter.length > 0) {
      result = result.filter((j) => statusFilter.includes(j.status))
    }
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter((j) =>
        (entityNames[j.id] || (j.metadata?.name as string) || j.id).toLowerCase().includes(q)
      )
    }
    return result
  }, [jobs, typeFilter, statusFilter, search, entityNames])

  function handleSelectJob(job: Job) {
    setSelectedJobId(job.id)
    setDetailOpen(true)
  }

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  return (
    <div className="space-y-6">
      <div data-tutorial="job-header" className="space-y-6">
        <PageHeader title="Jobs" description="Track your training and data generation runs" />

        <div className="animate-message-in rounded-xl border bg-card p-4">
          <div className="grid gap-3 md:grid-cols-3 mb-3">
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
                <Sparkles className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">SDG</p>
                <p className="text-xs text-muted-foreground">Generate synthetic training data</p>
              </div>
            </div>
            <div className="flex items-start gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                <GraduationCap className="h-3.5 w-3.5" />
              </div>
              <div>
                <p className="text-sm font-medium">Training</p>
                <p className="text-xs text-muted-foreground">Fine-tune a base model on your data</p>
              </div>
            </div>
          </div>
          <p className="text-xs text-muted-foreground border-t pt-3">
            Jobs are created from{" "}
            <Link to="/chat" className="inline-flex items-center gap-0.5 underline underline-offset-2 hover:text-foreground transition-colors">
              Chat <ArrowRight className="h-2.5 w-2.5" />
            </Link>{" "}
            or{" "}
            <Link to="/recipes" className="underline underline-offset-2 hover:text-foreground transition-colors">
              Recipes
            </Link>. Click any job to view real-time logs and metrics.
          </p>
        </div>
      </div>

      {isLoading ? (
        <TableSkeleton columns={6} />
      ) : jobs.length > 0 ? (
        <>
          <CostBreakdownChart />
          <div className="space-y-3">
            <p className="text-sm font-semibold text-muted-foreground">Filters</p>
            <div className="space-y-2.5">
              <FilterChips
                label="Type"
                options={TYPE_OPTIONS}
                selected={typeFilter}
                onChange={setTypeFilter}
              />
              <FilterChips
                label="Status"
                options={STATUS_OPTIONS}
                selected={statusFilter}
                onChange={setStatusFilter}
              />
            </div>
          </div>
          <SearchInput value={search} onChange={(v) => { setSearch(v); setPage(0) }} placeholder="Search jobs..." />
          <JobTable
            jobs={filteredJobs}
            page={page}
            onPageChange={setPage}
            onSelectJob={handleSelectJob}
          />
        </>
      ) : (
        <Empty className="border mt-4">
          <EmptyHeader>
            <EmptyMedia variant="icon"><Briefcase /></EmptyMedia>
            <EmptyTitle>No jobs yet</EmptyTitle>
            <EmptyDescription>Jobs appear here when you run training or SDG tasks.</EmptyDescription>
          </EmptyHeader>
          <EmptyContent>
            <Button asChild><Link to="/chat">Go to Chat</Link></Button>
          </EmptyContent>
        </Empty>
      )}

      <JobDetailPanel
        job={selectedJob}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </div>
  )
}
