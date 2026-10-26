import type { JsonSchema } from "../api/use-recipes"
import { formatLabel } from "../components/recipe-field-utils"

// Schema-driven required-field validation. The Pydantic-generated JSON schema
// (/api/v1/schemas) is the single source of truth for what's required; this walks
// a config against it and returns readable paths of required fields that are empty
// (missing / null / "" / [] / {}). Used to block Execute until they're filled.
// JSON-Schema `required` only checks presence, so we treat empty-as-missing here.

type Schema = Record<string, unknown>

function isEmpty(v: unknown): boolean {
  if (v === null || v === undefined) return true
  if (typeof v === "string") return v.trim() === ""
  if (Array.isArray(v)) return v.length === 0
  if (typeof v === "object") return Object.keys(v).length === 0
  return false // numbers (incl. 0) and booleans (incl. false) are filled
}

// Follow $ref into $defs and flatten a single-element allOf (Pydantic wraps a
// $ref + description that way). Guards against $ref cycles.
function resolveRef(
  schema: Schema,
  defs: Record<string, Schema>,
  seen = new Set<string>(),
): Schema {
  if (typeof schema.$ref === "string") {
    const name = schema.$ref.replace(/^#\/(?:\$defs|definitions)\//, "")
    if (seen.has(name)) return schema
    seen.add(name)
    return resolveRef((defs[name] as Schema) ?? {}, defs, seen)
  }
  if (Array.isArray(schema.allOf) && schema.allOf.length === 1) {
    return resolveRef(schema.allOf[0] as Schema, defs, seen)
  }
  return schema
}

function matchesConst(prop: Schema, value: unknown): boolean {
  if ("const" in prop) return prop.const === value
  if (Array.isArray(prop.enum)) return prop.enum.includes(value)
  return false
}

// Resolve a oneOf/anyOf node to the applicable branch: the sole non-null branch
// (Optional[X]), or the discriminated branch whose discriminator const matches the
// value. Returns null when it can't be resolved confidently — callers then skip,
// so we never emit a false positive that blocks a valid config.
function pickBranch(value: unknown, node: Schema, defs: Record<string, Schema>): Schema | null {
  const raw = (node.oneOf ?? node.anyOf) as Schema[] | undefined
  if (!raw) return null
  const branches = raw.map((b) => resolveRef(b, defs))
  const nonNull = branches.filter((b) => b.type !== "null")
  if (nonNull.length === 1) return nonNull[0] ?? null

  const disc = (node.discriminator as Schema | undefined)?.propertyName as string | undefined
  if (disc && value && typeof value === "object") {
    const dv = (value as Record<string, unknown>)[disc]
    for (const b of nonNull) {
      const prop = (b.properties as Record<string, Schema> | undefined)?.[disc]
      if (prop && matchesConst(prop, dv)) return b
    }
  }
  return null
}

// Join a raw path segment. Raw paths (e.g. "columns/0/model_alias") are what the
// form builds while rendering, so required markers can be looked up by position.
// Keep this in sync with the path building in recipe-config-form.tsx.
export function joinPath(path: string, seg: string | number): string {
  return path ? `${path}/${seg}` : String(seg)
}

// One required field, with a raw path (for markers) and a display path (for the
// "fill these" list) — so markers and the Submit-block derive from the same walk.
type RequiredField = { raw: string; display: string; empty: boolean }

function walk(
  value: unknown,
  schema: Schema | undefined,
  defs: Record<string, Schema>,
  raw: string,
  display: string,
  out: RequiredField[],
): void {
  if (!schema) return
  let node = resolveRef(schema, defs)
  if (node.oneOf || node.anyOf) {
    const picked = pickBranch(value, node, defs)
    if (!picked) return
    node = picked
  }

  const props = node.properties as Record<string, Schema> | undefined
  const required = node.required as string[] | undefined
  if (props || required) {
    const obj =
      value && typeof value === "object" && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : {}
    const displayFor = (key: string) =>
      display ? `${display} › ${formatLabel(key)}` : formatLabel(key)
    if (Array.isArray(required)) {
      for (const key of required) {
        out.push({ raw: joinPath(raw, key), display: displayFor(key), empty: isEmpty(obj[key]) })
      }
    }
    if (props) {
      for (const [key, childSchema] of Object.entries(props)) {
        if (key in obj && !isEmpty(obj[key])) {
          walk(obj[key], childSchema, defs, joinPath(raw, key), displayFor(key), out)
        }
      }
    }
  }

  const items = node.items as Schema | undefined
  if (items && Array.isArray(value)) {
    value.forEach((item, i) => walk(item, items, defs, joinPath(raw, i), display, out))
  }
}

function walkAll(
  config: unknown,
  typeSchema: JsonSchema | null | undefined,
  requiredKeys: string[],
): RequiredField[] {
  const out: RequiredField[] = []
  if (typeSchema) {
    const defs = ((typeSchema.$defs ?? typeSchema.definitions) ?? {}) as Record<string, Schema>
    walk(config, typeSchema as Schema, defs, "", "", out)
  }
  // Per-recipe required top-level fields (declared on the recipe, not the shared
  // schema — e.g. Knowledge Ingestion requires document_ids).
  const obj =
    config && typeof config === "object" && !Array.isArray(config)
      ? (config as Record<string, unknown>)
      : {}
  for (const key of requiredKeys) {
    out.push({ raw: key, display: requiredKeyLabel(key), empty: isEmpty(obj[key]) })
  }
  return out
}

// The dataset-reference keys render as a single "Dataset" picker, so show "Dataset"
// in the block list rather than the raw field name ("Parent Job Id").
const DATASET_REF_KEYS = new Set(["parent_job_id", "data_run_id", "eval_data_run_id"])
function requiredKeyLabel(key: string): string {
  return DATASET_REF_KEYS.has(key) ? "Dataset" : formatLabel(key)
}

// Readable paths of required fields that are empty — blocks Submit until filled.
export function collectMissingRequired(
  config: unknown,
  typeSchema?: JsonSchema | null,
  requiredKeys: string[] = [],
): string[] {
  return [...new Set(walkAll(config, typeSchema, requiredKeys).filter((f) => f.empty).map((f) => f.display))]
}

// Raw paths of ALL required fields (branch-aware) — drives the form's required
// markers so they match exactly what blocks Submit (no name-keyed false positives).
export function collectRequiredPaths(
  config: unknown,
  typeSchema?: JsonSchema | null,
  requiredKeys: string[] = [],
): Set<string> {
  return new Set(walkAll(config, typeSchema, requiredKeys).map((f) => f.raw))
}
