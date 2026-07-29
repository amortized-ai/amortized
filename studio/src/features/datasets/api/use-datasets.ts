import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import {
  searchMlflowExperiments,
  searchMlflowRuns,
  getMlflowRun,
  getMlflowArtifactContent,
  listArtifacts,
  getArtifactJson,
  deleteDataset,
} from "@/lib/api-client"
import type { DatasetRecord, DatasetSample, MlflowRun } from "@/types/api"

function runToDataset(run: MlflowRun): DatasetRecord {
  const tags: Record<string, string> = {}
  for (const t of run.data?.tags ?? []) tags[t.key] = t.value

  const params: Record<string, string> = {}
  for (const p of run.data?.params ?? []) params[p.key] = p.value

  const metrics: Record<string, number> = {}
  for (const m of run.data?.metrics ?? []) metrics[m.key] = m.value

  return {
    run_id: run.info.run_id,
    name: tags["dataset_name"] ?? run.info.run_name ?? run.info.run_id,
    run_name: run.info.run_name,
    experiment_id: run.info.experiment_id,
    artifact_uri: run.info.artifact_uri,
    created_at: run.info.start_time,
    metrics,
    params,
    tags,
  }
}

export function useDatasets() {
  return useQuery<DatasetRecord[]>({
    queryKey: ["mlflow", "datasets"],
    queryFn: async () => {
      const exps = await searchMlflowExperiments()
      const ids = (exps.experiments ?? []).map((e) => e.experiment_id)
      if (ids.length === 0) return []
      const resp = await searchMlflowRuns({
        experiment_ids: ids,
        filter_string: "tags.job_type = 'sdg'",
        order_by: ["start_time DESC"],
        max_results: 100,
      })
      return (resp.runs ?? []).map(runToDataset)
    },
  })
}

export function useDataset(runId: string | null) {
  return useQuery<DatasetRecord>({
    queryKey: ["mlflow", "datasets", runId],
    queryFn: async () => {
      const resp = await getMlflowRun(runId!)
      return runToDataset(resp.run)
    },
    enabled: !!runId,
  })
}

export function useDeleteDataset() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (runId: string) => deleteDataset(runId),
    onSuccess: () => {
      toast.success("Dataset deleted successfully")
    },
    onError: (err) => {
      toast.error(`Failed to delete dataset: ${err instanceof Error ? err.message : "Unknown error"}`)
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["mlflow", "datasets"] })
    },
  })
}

function parseRecord(
  parsed: Record<string, unknown>,
  index: number,
): DatasetSample {
  let messages: DatasetSample["messages"] = []
  const metadata: Record<string, unknown> = {}

  if (Array.isArray(parsed.messages)) {
    messages = parsed.messages as DatasetSample["messages"]
  } else if (typeof parsed.messages === "string") {
    const str = parsed.messages as string
    const jsonMatch = str.match(/\[[\s\S]*\]/)
    if (jsonMatch) {
      try {
        const arr = JSON.parse(jsonMatch[0])
        if (Array.isArray(arr)) messages = arr as DatasetSample["messages"]
      } catch { /* unparseable */ }
    }
  } else {
    for (const [, value] of Object.entries(parsed)) {
      if (
        value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        Array.isArray((value as Record<string, unknown>).messages)
      ) {
        messages = (value as Record<string, unknown>)
          .messages as DatasetSample["messages"]
        break
      }
    }
  }

  messages = messages.map((m) => ({
    ...m,
    content:
      typeof m.content === "string"
        ? m.content
        : m.content != null
          ? JSON.stringify(m.content, null, 2)
          : "",
  }))

  for (const [key, value] of Object.entries(parsed)) {
    if (key !== "messages") metadata[key] = value
  }

  return { index, messages, metadata }
}

/**
 * Fetch data samples from a dataset's artifacts.
 * Supports both parquet (Data Designer) and JSONL (legacy) formats.
 */
export function useDatasetSamples(
  experimentId: string | null,
  runId: string | null,
) {
  return useQuery<DatasetSample[]>({
    queryKey: ["mlflow", "datasets", runId, "samples"],
    queryFn: async () => {
      const { files } = await listArtifacts(
        experimentId!,
        runId!,
        "generated_data",
      )

      const parquet = files.find((f) => f.path.endsWith(".parquet"))
      if (parquet) {
        const records = await getArtifactJson(
          experimentId!,
          runId!,
          parquet.path,
        )
        return records.map((r, i) => parseRecord(r, i))
      }

      const jsonl = files.find((f) => f.path.endsWith(".jsonl"))
      if (jsonl) {
        const raw = await getMlflowArtifactContent(
          experimentId!,
          runId!,
          jsonl.path,
        )
        const lines = raw.trim().split("\n").filter(Boolean)
        return lines.map((line, i) => {
          const parsed = JSON.parse(line) as Record<string, unknown>
          return parseRecord(parsed, i)
        })
      }

      return []
    },
    enabled: !!experimentId && !!runId,
  })
}
