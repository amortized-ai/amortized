// Canonical placeholder tokens that starter templates
// (agents/*/skills/**/reference-payload.json) ship for the agent to resolve via
// its tools (list_documents, list_models) or for lineage chaining. The recipe
// form treats them as "unset" so a picker prompts for a real value instead of
// showing a bogus selection.
//
// Mirror of the backend's src/amortized/core/placeholders.py — keep the two in sync.

export const PLACEHOLDER_DOCUMENT_ID = "<doc-id-from-list_documents>"
export const PLACEHOLDER_MODEL = "<from-list_models>"
export const PLACEHOLDER_PARENT_JOB_ID = "<SDG_JOB_ID>"

export const PLACEHOLDER_TOKENS: ReadonlySet<string> = new Set([
  PLACEHOLDER_DOCUMENT_ID,
  PLACEHOLDER_MODEL,
  PLACEHOLDER_PARENT_JOB_ID,
])

export function isPlaceholder(value: string): boolean {
  return PLACEHOLDER_TOKENS.has(value)
}

// Remove placeholder tokens from a config so they are never carried as values:
// placeholder strings become "" (keeps the field so its picker still renders),
// placeholder entries are dropped from arrays, and nested objects/arrays are
// walked. Applied when a starter template / recipe is loaded into the form.
export function stripPlaceholders<T>(value: T): T {
  if (typeof value === "string") {
    return (isPlaceholder(value) ? "" : value) as T
  }
  if (Array.isArray(value)) {
    return value
      .filter((v) => !(typeof v === "string" && isPlaceholder(v)))
      .map(stripPlaceholders) as T
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, stripPlaceholders(v)]),
    ) as T
  }
  return value
}
