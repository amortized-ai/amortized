import type { PromptView } from "../types"

interface McpContentBlock {
  type: string
  text?: string
}

function extractPromptViews(raw: unknown): PromptView[] {
  if (!Array.isArray(raw)) return []
  const out: PromptView[] = []
  for (const item of raw) {
    if (!item || typeof item !== "object") continue
    const o = item as Record<string, unknown>
    if (typeof o.text === "string" && o.text.trim()) {
      out.push({
        role: typeof o.role === "string" ? o.role : "",
        label: typeof o.label === "string" ? o.label : "",
        column: typeof o.column === "string" ? o.column : "",
        text: o.text,
      })
    }
  }
  return out
}

export function unwrapToolResult(raw: string): unknown {
  if (!raw) return null

  let parsed: unknown
  try {
    parsed = typeof raw === "string" ? JSON.parse(raw) : raw
  } catch {
    return null
  }

  if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
    return parsed
  }

  if (Array.isArray(parsed)) {
    const textBlock = parsed.find(
      (b: unknown): b is McpContentBlock =>
        typeof b === "object" && b !== null && (b as McpContentBlock).type === "text" && typeof (b as McpContentBlock).text === "string",
    )
    if (textBlock?.text) {
      try {
        return JSON.parse(textBlock.text)
      } catch {
        return null
      }
    }
  }

  return null
}

const UUID_RE = /[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/i

export function extractJobInfo(result: string): { jobId: string | null; jobType: string } {
  const parsed = unwrapToolResult(result)

  if (parsed && typeof parsed === "object") {
    const obj = parsed as Record<string, unknown>
    if (obj.dry_run) return { jobId: null, jobType: "SDG" }
    const id = typeof obj.id === "string" ? obj.id : null
    const type = typeof obj.type === "string" ? obj.type.toUpperCase() : "SDG"
    if (id) return { jobId: id, jobType: type }
  }

  const match = result.match(UUID_RE)
  if (match) return { jobId: match[0], jobType: "SDG" }

  return { jobId: null, jobType: "SDG" }
}

export interface ValidatedJobConfig {
  valid: boolean
  jobType: string
  config: Record<string, unknown>
  parentJobId: string
  recipe: string
  warnings: string[]
  dataRecordCount: number | null
  assessorPrompt: string | null
  prompts: PromptView[]
}

export function extractValidatedJobConfig(result: string): ValidatedJobConfig | null {
  const parsed = unwrapToolResult(result)
  if (!parsed || typeof parsed !== "object") return null
  const obj = parsed as Record<string, unknown>
  if (obj.valid === true && typeof obj.job_type === "string" && !obj.id) {
    return {
      valid: true,
      jobType: obj.job_type as string,
      config: (obj.config as Record<string, unknown>) ?? {},
      parentJobId: (obj.parent_job_id as string) ?? "",
      recipe: (obj.recipe as string) ?? "",
      warnings: (obj.warnings as string[]) ?? [],
      dataRecordCount:
        typeof obj.data_record_count === "number" ? obj.data_record_count : null,
      assessorPrompt:
        typeof obj.assessor_prompt === "string" ? obj.assessor_prompt : null,
      prompts: extractPromptViews(obj.prompts),
    }
  }
  return null
}

export const VALIDATE_TO_CREATE_ENDPOINT: Record<string, string> = {
  validate_sdg_job: "/api/v1/jobs/sdg",
  validate_training_job: "/api/v1/jobs/training",
  validate_eval_job: "/api/v1/jobs/eval",
  validate_recipe_job: "/api/v1/jobs/recipe",
}
