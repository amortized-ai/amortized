import type { JsonSchema } from "../api/use-recipes"

// Client-side pre-submit check: which of the schema's required top-level fields
// are missing or empty in the config. The backend stays the source of truth (and
// now surfaces field-level errors on 422), so this just catches the common cases
// early with instant feedback instead of a round-trip.
export function missingRequiredFields(
  config: Record<string, unknown>,
  schema: JsonSchema | undefined | null,
): string[] {
  if (!schema || typeof schema !== "object") return []
  const required = Array.isArray(schema.required) ? schema.required : []
  const missing: string[] = []
  for (const field of required) {
    if (typeof field !== "string") continue
    const v = config[field]
    const empty =
      v == null ||
      (typeof v === "string" && v.trim() === "") ||
      (Array.isArray(v) && v.length === 0) ||
      (typeof v === "object" && !Array.isArray(v) && Object.keys(v as object).length === 0)
    if (empty) missing.push(field)
  }
  return missing
}
