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
