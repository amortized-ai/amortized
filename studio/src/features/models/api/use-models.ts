import { useQuery } from "@tanstack/react-query"
import {
  searchMlflowRegisteredModels,
  searchMlflowModelVersions,
  getMlflowRun,
  getMlflowMetricHistory,
  getJobs,
} from "@/lib/api-client"
import type { ModelRecord, MlflowRegisteredModel, MlflowRun, Job } from "@/types/api"

function registeredModelToRecord(m: MlflowRegisteredModel): ModelRecord {
  const latest = m.latest_versions?.[0]
  const tags: Record<string, string> = {}
  if (m.tags) {
    for (const t of m.tags) tags[t.key] = t.value
  }

  return {
    name: m.name,
    version: latest?.version ?? "0",
    run_id: latest?.run_id ?? "",
    source: latest?.source ?? "",
    created_at: m.creation_timestamp,
    description: m.description ?? "",
    aliases: latest?.aliases ?? [],
    tags,
  }
}

export function useModels() {
  return useQuery<ModelRecord[]>({
    queryKey: ["mlflow", "models"],
    queryFn: async () => {
      const resp = await searchMlflowRegisteredModels()
      return (resp.registered_models ?? []).map(registeredModelToRecord)
    },
  })
}

export function useModel(name: string | null) {
  return useQuery<ModelRecord[]>({
    queryKey: ["mlflow", "models", name, "versions"],
    queryFn: async () => {
      const resp = await searchMlflowModelVersions(`name='${name}'`)
      return (resp.model_versions ?? []).map((v) => {
        const tags: Record<string, string> = {}
        if (v.tags) {
          for (const t of v.tags) tags[t.key] = t.value
        }
        return {
          name: v.name,
          version: v.version,
          run_id: v.run_id,
          source: v.source,
          created_at: v.creation_timestamp,
          description: "",
          aliases: v.aliases ?? [],
          tags,
        }
      })
    },
    enabled: !!name,
  })
}

export interface ModelRunData {
  run: MlflowRun
  params: Record<string, string>
  finalMetrics: Record<string, number>
  tags: Record<string, string>
  histories: Record<string, Array<{ step: number; value: number }>>
}

const HISTORY_KEYS = ["loss", "train_loss", "grad_norm", "learning_rate", "epoch"]

export function useModelRunData(runId: string | null) {
  return useQuery<ModelRunData | null>({
    queryKey: ["mlflow", "model-run", runId],
    queryFn: async () => {
      if (!runId) return null
      const { run } = await getMlflowRun(runId)

      const params: Record<string, string> = {}
      for (const p of run.data.params ?? []) params[p.key] = p.value

      const finalMetrics: Record<string, number> = {}
      for (const m of run.data.metrics ?? []) finalMetrics[m.key] = m.value

      const tags: Record<string, string> = {}
      for (const t of run.data.tags ?? []) tags[t.key] = t.value

      const available = new Set((run.data.metrics ?? []).map((m) => m.key))
      const keysToFetch = HISTORY_KEYS.filter((k) => available.has(k))

      const historyResults = await Promise.all(
        keysToFetch.map(async (key) => {
          const resp = await getMlflowMetricHistory(runId, key)
          const entries = (resp.metrics ?? [])
            .map((m) => ({ step: m.step, value: m.value }))
            .sort((a, b) => a.step - b.step)
          return [key, entries] as const
        }),
      )

      const histories: Record<string, Array<{ step: number; value: number }>> = {}
      for (const [key, entries] of historyResults) {
        histories[key] = entries
      }

      // Normalize: if "loss" is empty but "train_loss" has data, copy it over
      if ((!histories["loss"] || histories["loss"].length === 0) && histories["train_loss"]?.length) {
        histories["loss"] = histories["train_loss"]
      }

      return { run, params, finalMetrics, tags, histories }
    },
    enabled: !!runId,
  })
}

export interface ModelJobsData {
  trainingJob: Job | null
  sdgJob: Job | null
  evalJobs: Job[]
}

export function useModelJobs(runId: string | null) {
  return useQuery<ModelJobsData>({
    queryKey: ["model-jobs", runId],
    queryFn: async () => {
      const result: ModelJobsData = { trainingJob: null, sdgJob: null, evalJobs: [] }
      if (!runId) return result

      try {
        const allJobs = await getJobs()
        const training = allJobs.find(
          (j) => j.type === "training" && j.mlflow_run_id === runId,
        )
        if (training) {
          result.trainingJob = training
          if (training.parent_job_id) {
            const parent = allJobs.find((j) => j.id === training.parent_job_id)
            if (parent && parent.type === "sdg") {
              result.sdgJob = parent
            }
          }
          result.evalJobs = []
        }
      } catch {
        // Jobs API may not be available
      }
      return result
    },
    enabled: !!runId,
  })
}
