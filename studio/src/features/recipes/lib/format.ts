import type { Job } from "@/types/api"

export function formatRecipeType(type: string): string {
  const map: Record<string, string> = {
    training: "Training",
    sdg: "SDG",
  }
  return map[type.toLowerCase()] ?? type
}

export function recipeTypeClassName(type: string): string {
  switch (type.toLowerCase()) {
    case "sdg":
      return "border-transparent bg-[#ece6ff] text-[#5e40be] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]"
    case "training":
      return "border-transparent bg-[#e0f0ff] text-[#0066cc] dark:bg-[#003366]/40 dark:text-[#4394e5]"
    default:
      return ""
  }
}

const ALGORITHM_LABELS: Record<string, string> = {
  sft: "SFT",
  lora_sft: "LoRA SFT",
  osft: "OSFT",
  dpo: "DPO",
  grpo: "GRPO",
  lora_grpo: "LoRA GRPO",
  kto: "KTO",
  gkd: "GKD",
  gepa: "GEPA",
}

export function formatAlgorithm(algo: string): string {
  return ALGORITHM_LABELS[algo] || algo
}

export function statusClassName(status: string): string {
  switch (status) {
    case "succeeded":
      return "border-transparent bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400"
    case "failed":
      return "border-transparent bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
    case "running":
      return "border-transparent bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400"
    case "queued":
    case "provisioning":
      return "border-transparent bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400"
    case "cancelled":
      return "border-transparent bg-gray-100 text-gray-600 dark:bg-gray-800/30 dark:text-gray-400"
    default:
      return ""
  }
}

// --- Recipe entry: deduped job with dataset-aware naming ---

export interface DatasetInfo {
  name: string
  topic: string
  samples: string
}

export interface RecipeEntry {
  job: Job
  name: string
  description: string
  version: string
}

function sortDeep(obj: unknown): unknown {
  if (Array.isArray(obj)) return obj.map(sortDeep)
  if (obj && typeof obj === "object") {
    const record = obj as Record<string, unknown>
    return Object.keys(record)
      .sort()
      .reduce<Record<string, unknown>>((acc, k) => {
        acc[k] = sortDeep(record[k])
        return acc
      }, {})
  }
  return obj
}

function configFingerprint(config: Record<string, unknown>): string {
  return JSON.stringify(sortDeep(config))
}

function deriveBaseName(
  job: Job,
  datasetMap: Map<string, DatasetInfo>,
): string {
  const config = job.config ?? {}
  const typeSuffix = job.type === "training" ? "train" : "sdg"

  const ds =
    job.type === "training" && job.parent_job_id
      ? datasetMap.get(job.parent_job_id)
      : datasetMap.get(job.id)

  if (ds?.name && ds?.topic) return `${ds.name} - ${ds.topic} (${typeSuffix})`
  if (ds?.topic) return `${ds.topic} (${typeSuffix})`
  if (ds?.name) return `${ds.name} (${typeSuffix})`

  if (job.type === "training") {
    const algorithm = (config.algorithm as string) || ""
    const model = (config.model_name_or_path as string) || (config.model_path as string) || ""
    const modelShort = model.split("/").pop() || model
    if (algorithm && modelShort) return `${formatAlgorithm(algorithm)} - ${modelShort} (train)`
    return "Training"
  }

  const numRecords = config.num_records as number | undefined
  if (numRecords) return `Data Generation - ${numRecords} samples (sdg)`
  return "Data Generation (sdg)"
}

function deriveDescription(job: Job, datasetMap: Map<string, DatasetInfo>): string {
  const config = job.config ?? {}

  if (job.type === "training") {
    const parts: string[] = []
    const model =
      (config.model_name_or_path as string) ||
      (config.model_path as string) ||
      ""
    if (model) parts.push(model)
    const lr = config.learning_rate as number | undefined
    if (lr) parts.push(`lr=${lr}`)
    const epochs =
      (config.num_train_epochs as number | undefined) ||
      (config.num_epochs as number | undefined)
    if (epochs) parts.push(`${epochs} epochs`)
    return parts.join(", ")
  }

  if (job.type === "sdg") {
    const ds = datasetMap.get(job.id)
    const parts: string[] = []
    const modelConfigs = config.model_configs as Array<{ model?: string }> | undefined
    if (modelConfigs?.length) parts.push(`Teacher: ${modelConfigs[0]?.model || "unknown"}`)
    const numRecords = config.num_records as number | undefined
    if (numRecords) parts.push(`${numRecords} samples`)
    if (ds?.name) parts.push(ds.name)
    return parts.join(", ")
  }

  return ""
}

export function buildRecipeEntries(
  jobs: Job[],
  datasetMap: Map<string, DatasetInfo>,
): RecipeEntry[] {
  const sorted = [...jobs].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
  )

  const seen = new Map<string, Job>()
  const deduped: Job[] = []
  for (const job of sorted) {
    const fp = configFingerprint(job.config ?? {})
    if (!seen.has(fp)) {
      seen.set(fp, job)
      deduped.push(job)
    }
  }

  const baseNames = deduped.map((j) => deriveBaseName(j, datasetMap))
  const nameCounts = new Map<string, number>()
  for (const bn of baseNames) {
    nameCounts.set(bn, (nameCounts.get(bn) ?? 0) + 1)
  }

  const nameCounters = new Map<string, number>()
  const entries: RecipeEntry[] = []
  for (let i = 0; i < deduped.length; i++) {
    const job = deduped[i]!
    const baseName = baseNames[i]!
    const needsVersion = (nameCounts.get(baseName) ?? 0) > 1

    let version = ""
    if (needsVersion) {
      const n = (nameCounters.get(baseName) ?? 0) + 1
      nameCounters.set(baseName, n)
      version = `v${n}`
    }

    entries.push({
      job,
      name: needsVersion ? `${baseName} ${version}` : baseName,
      description: deriveDescription(job, datasetMap),
      version,
    })
  }

  return entries.reverse()
}
