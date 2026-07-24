import {
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import { toast } from "sonner"
import {
  getJobs,
  getJob,
  cancelJob,
  getJobLogs,
  getMlflowRun,
  getMlflowMetricHistory,
} from "@/lib/api-client"
import type { Job, JobFilters, PaginationParams, MlflowRun } from "@/types/api"

const ACTIVE_STATUSES = new Set(["queued", "provisioning", "running"])

export function useJobs(filters?: JobFilters, pagination?: PaginationParams) {
  return useQuery<Job[]>({
    queryKey: ["jobs", filters, pagination],
    queryFn: () => getJobs(filters, pagination),
    refetchInterval: (query) => {
      const hasActive = query.state.data?.some((j) => ACTIVE_STATUSES.has(j.status))
      return hasActive ? 5000 : false
    },
  })
}

export function useJob(id: string | null) {
  return useQuery<Job>({
    queryKey: ["jobs", id],
    queryFn: () => getJob(id!),
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status && ACTIVE_STATUSES.has(status)) return 3000
      return false
    },
  })
}

export function useCancelJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onMutate: async (jobId) => {
      await queryClient.cancelQueries({ queryKey: ["jobs"] })

      const previousJobs = queryClient.getQueryData<Job[]>(["jobs"])

      queryClient.setQueryData<Job[]>(
        ["jobs"],
        (old) =>
          Array.isArray(old)
            ? old.map((j) =>
                j.id === jobId ? { ...j, status: "cancelled" as const } : j,
              )
            : old,
      )

      return { previousJobs }
    },
    onSuccess: (_data, jobId) => {
      toast.success(`Job ${jobId.slice(0, 8)} cancelled successfully`)
    },
    onError: (err, _jobId, context) => {
      toast.error(`Failed to cancel job: ${err instanceof Error ? err.message : "Unknown error"}`)
      if (context?.previousJobs) {
        queryClient.setQueryData<Job[]>(["jobs"], context.previousJobs)
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
  })
}

export function useJobLogs(jobId: string | null, isActive = false) {
  return useQuery<string[]>({
    queryKey: ["jobs", jobId, "logs"],
    queryFn: () => getJobLogs(jobId!),
    enabled: !!jobId,
    refetchInterval: isActive ? 5000 : false,
  })
}

export function useJobMlflowRun(runId: string | null) {
  return useQuery<MlflowRun>({
    queryKey: ["mlflow", "runs", runId],
    queryFn: async () => {
      const resp = await getMlflowRun(runId!)
      return resp.run
    },
    enabled: !!runId,
  })
}

/** History for a single metric key: step-ordered (step, value) pairs. */
export interface MetricHistory {
  step: number
  value: number
}

/** Aggregated metrics data returned by useJobMlflowMetrics. */
export interface AllMetricsData {
  /** Per-key time-series histories (only populated for keys that exist). */
  histories: Record<string, MetricHistory[]>
  /** Final/latest value for every metric key the run logged. */
  finalValues: Record<string, number>
  /** Run params (e.g. model_name_or_path, num_train_epochs, …). */
  params: Record<string, string>
}

/** Metric keys we fetch full step-level history for. */
const HISTORY_KEYS = ["loss", "train_loss", "grad_norm", "learning_rate", "epoch"]

export function useJobMlflowMetrics(runId: string | null) {
  return useQuery<AllMetricsData>({
    queryKey: ["mlflow", "metrics", runId, "all"],
    queryFn: async () => {
      // 1. Fetch the run to discover which metrics were logged
      const { run } = await getMlflowRun(runId!)

      const finalValues: Record<string, number> = {}
      for (const m of run.data.metrics) {
        finalValues[m.key] = m.value
      }

      const params: Record<string, string> = {}
      for (const p of run.data.params) {
        params[p.key] = p.value
      }

      // 2. Fetch step-level history for each relevant key (in parallel)
      const available = new Set(run.data.metrics.map((m) => m.key))
      const keysToFetch = HISTORY_KEYS.filter((k) => available.has(k))

      const results = await Promise.all(
        keysToFetch.map(async (key) => {
          const resp = await getMlflowMetricHistory(runId!, key)
          return { key, entries: resp.metrics ?? [] }
        }),
      )

      const histories: Record<string, MetricHistory[]> = {}
      for (const { key, entries } of results) {
        histories[key] = entries
          .map((e) => ({ step: e.step, value: e.value }))
          .sort((a, b) => a.step - b.step)
      }

      return { histories, finalValues, params }
    },
    enabled: !!runId,
  })
}
