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
