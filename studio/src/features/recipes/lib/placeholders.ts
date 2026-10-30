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

// Prompt-store references (#541). A reviewed prompt is stored once and cited from
// a config by an unguessable id as the WHOLE value of a prompt field — so, like the
// backend's core.placeholders, matching and substitution are full-value.
const PROMPT_REF_RE = /^<prompt:(prm_[A-Za-z0-9]+)>$/

export function makePromptRef(promptId: string): string {
  return `<prompt:${promptId}>`
}

export function promptRefId(value: unknown): string | null {
  if (typeof value !== "string") return null
  const match = PROMPT_REF_RE.exec(value)
  return match?.[1] ?? null
}

export function isPromptRef(value: unknown): boolean {
  return promptRefId(value) !== null
}

// Distinct prompt ids referenced anywhere in a config subtree.
export function findPromptRefs(value: unknown): string[] {
  const found = new Set<string>()
  const walk = (v: unknown): void => {
    const ref = promptRefId(v)
    if (ref !== null) {
      found.add(ref)
    } else if (Array.isArray(v)) {
      v.forEach(walk)
    } else if (v && typeof v === "object") {
      Object.values(v).forEach(walk)
    }
  }
  walk(value)
  return [...found]
}

// Return a copy of `value` with every <prompt:...> ref replaced by its text from
// `texts` (id -> text). A ref with no entry is left as-is. Pure (does not mutate).
export function resolvePromptRefs<T>(value: T, texts: Record<string, string>): T {
  const walk = (v: unknown): unknown => {
    const ref = promptRefId(v)
    if (ref !== null) return ref in texts ? texts[ref] : v
    if (Array.isArray(v)) return v.map(walk)
    if (v && typeof v === "object") {
      return Object.fromEntries(Object.entries(v).map(([k, val]) => [k, walk(val)]))
    }
    return v
  }
  return walk(value) as T
}

// Remove placeholder tokens so they are never carried as values: placeholder
// strings become "" (the field stays, so its picker still renders), placeholder
// entries are dropped from arrays, and nested objects/arrays are walked. Applied
// when a starter template / recipe is loaded into the form. Placeholders are only
// agent-resolution hints — not config values and not a required-ness signal.
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
