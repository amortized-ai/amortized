interface McpContentBlock {
  type: string
  text?: string
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
    }
  }
  return null
}

export const VALIDATE_TO_CREATE_ENDPOINT: Record<string, string> = {
  validate_sdg_job: "/api/v1/jobs/sdg",
  validate_training_job: "/api/v1/jobs/training",
  validate_recipe_job: "/api/v1/jobs/recipe",
}
