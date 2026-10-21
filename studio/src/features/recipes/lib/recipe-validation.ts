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
  if (typeof v === "string") return v === ""
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

function walk(
  value: unknown,
  schema: Schema | undefined,
  defs: Record<string, Schema>,
  path: string,
  acc: string[],
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
    if (Array.isArray(required)) {
      for (const key of required) {
        if (isEmpty(obj[key])) {
          acc.push(path ? `${path} › ${formatLabel(key)}` : formatLabel(key))
        }
      }
    }
    if (props) {
      for (const [key, childSchema] of Object.entries(props)) {
        if (key in obj && !isEmpty(obj[key])) {
          const childPath = path ? `${path} › ${formatLabel(key)}` : formatLabel(key)
          walk(obj[key], childSchema, defs, childPath, acc)
        }
      }
    }
  }

  const items = node.items as Schema | undefined
  if (items && Array.isArray(value)) {
    value.forEach((item, i) => walk(item, items, defs, `${path}[${i}]`, acc))
  }
}

export function collectMissingRequired(config: unknown, typeSchema?: JsonSchema | null): string[] {
  if (!typeSchema) return []
  const defs = ((typeSchema.$defs ?? typeSchema.definitions) ?? {}) as Record<string, Schema>
  const acc: string[] = []
  walk(config, typeSchema as Schema, defs, "", acc)
  return [...new Set(acc)]
}
