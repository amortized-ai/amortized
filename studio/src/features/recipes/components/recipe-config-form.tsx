import { useState } from "react"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ChevronDown } from "lucide-react"
import type { JobType } from "@/types/api"
import type { JsonSchema } from "../api/use-recipes"
import { FieldLabel, FieldDescription } from "./recipe-fields"
import { formatLabel, SdgModelAliasesContext, type FieldMeta } from "./recipe-field-utils"
import {
  DocumentPickerField,
  DatasetPickerField,
  ModelPickerField,
  ModelAliasField,
} from "./recipe-pickers"
import { collectRequiredPaths, joinPath } from "../lib/recipe-validation"

interface RecipeConfigFormProps {
  type: JobType
  config: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
  schema?: JsonSchema | null
  // Per-recipe required top-level fields (declared on the recipe, not the shared
  // schema — e.g. Knowledge Ingestion requires document_ids).
  requiredKeys?: string[]
}

// --- Schema helpers ---
// Flattens all $defs into a fieldName -> {description, enum, required} map so
// descriptions, enums and required markers are available at any nesting depth,
// not just top-level.

const FALLBACK_DESCRIPTIONS: Record<string, string> = {
  drop: "Exclude this column from the final dataset output",
  propagate_skip: "Auto-skip this column when any of its dependencies were skipped",
  conditional_params: "Conditional parameters — keys are conditions, values are params to use when met",
  column_type: "Column type (sampler, llm-text, llm-code, expression, etc.)",
  sampler_type: "Sampling strategy (category, uniform, etc.)",
  with_trace: "Include generation trace in output (none, full)",
}

function buildFieldDescriptionMap(schema: JsonSchema): Map<string, FieldMeta> {
  const map = new Map<string, FieldMeta>()
  const defs = (schema.$defs ?? schema.definitions ?? {}) as Record<string, JsonSchema>

  // A field name is treated as required if it appears in any schema's `required`
  // array (top-level or any $def). Name-keyed, so acceptable for v1.
  const requiredNames = new Set<string>()
  function collectRequired(s: JsonSchema) {
    const req = s.required as string[] | undefined
    if (Array.isArray(req)) for (const r of req) requiredNames.add(r)
  }
  collectRequired(schema)
  for (const def of Object.values(defs)) collectRequired(def)

  function extractFromProps(props: Record<string, JsonSchema>) {
    for (const [key, fieldSchema] of Object.entries(props)) {
      if (map.has(key)) continue
      const desc = fieldSchema.description as string | undefined
      const enumVals = fieldSchema.enum as string[] | undefined
      const required = requiredNames.has(key)
      if (desc || enumVals || required) {
        map.set(key, { description: desc, enumValues: enumVals, required })
      }
    }
  }

  if (schema.properties) {
    extractFromProps(schema.properties as Record<string, JsonSchema>)
  }

  for (const def of Object.values(defs)) {
    if (def.properties) {
      extractFromProps(def.properties as Record<string, JsonSchema>)
    }
  }

  return map
}

let _cachedSchemaMap: Map<string, FieldMeta> | null = null
let _cachedSchemaRef: JsonSchema | null = null

function getFieldMeta(fieldKey: string, schema?: JsonSchema | null): FieldMeta {
  if (!schema) {
    const fb = FALLBACK_DESCRIPTIONS[fieldKey]
    return fb ? { description: fb } : {}
  }

  if (schema !== _cachedSchemaRef) {
    _cachedSchemaMap = buildFieldDescriptionMap(schema)
    _cachedSchemaRef = schema
  }

  const meta = _cachedSchemaMap!.get(fieldKey)
  if (meta) return meta

  const fb = FALLBACK_DESCRIPTIONS[fieldKey]
  return fb ? { description: fb } : {}
}

// --- Primitives ---

function NumberField({
  label,
  value,
  onChange,
  step,
  description,
  required,
}: {
  label: string
  value: number | undefined
  onChange: (v: number | undefined) => void
  step?: string
  description?: string
  required?: boolean
}) {
  const id = `field-${label}`
  return (
    <div className="space-y-1">
      <FieldLabel label={label} htmlFor={id} required={required} />
      <Input
        id={id}
        type="number"
        step={step}
        value={value ?? ""}
        onChange={(e) => {
          const raw = e.target.value
          onChange(raw === "" ? undefined : Number(raw))
        }}
        className="h-9"
      />
      <FieldDescription text={description} />
    </div>
  )
}

function TextField({
  label,
  value,
  onChange,
  description,
  required,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  description?: string
  required?: boolean
}) {
  const id = `field-${label}`
  return (
    <div className="space-y-1">
      <FieldLabel label={label} htmlFor={id} required={required} />
      <Input id={id} value={value} onChange={(e) => onChange(e.target.value)} className="h-9" />
      <FieldDescription text={description} />
    </div>
  )
}

function LongTextField({
  label,
  value,
  onChange,
  description,
  required,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  description?: string
  required?: boolean
}) {
  return (
    <div className="space-y-1">
      <FieldLabel label={label} required={required} />
      <Textarea value={value} onChange={(e) => onChange(e.target.value)} className="min-h-[80px] text-sm" />
      <FieldDescription text={description} />
    </div>
  )
}

function BooleanField({
  label,
  value,
  onChange,
  description,
  required,
}: {
  label: string
  value: boolean
  onChange: (v: boolean) => void
  description?: string
  required?: boolean
}) {
  const id = `field-${label}`
  return (
    <div>
      <div className="flex items-center justify-between">
        <FieldLabel label={label} htmlFor={id} required={required} />
        <Switch id={id} checked={value} onCheckedChange={onChange} />
      </div>
      <FieldDescription text={description} />
    </div>
  )
}

function EnumField({
  label,
  value,
  options,
  onChange,
  description,
  required,
}: {
  label: string
  value: string
  options: string[]
  onChange: (v: string) => void
  description?: string
  required?: boolean
}) {
  return (
    <div className="space-y-1">
      <FieldLabel label={label} required={required} />
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o} value={o}>
              {o.replace(/_/g, " ").toUpperCase()}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription text={description} />
    </div>
  )
}

function JsonField({
  label,
  value,
  onChange,
  required,
}: {
  label: string
  value: unknown
  onChange: (v: unknown) => void
  required?: boolean
}) {
  const [text, setText] = useState(() => JSON.stringify(value, null, 2))
  const [error, setError] = useState<string | null>(null)

  function handleChange(raw: string) {
    setText(raw)
    try {
      setError(null)
      onChange(JSON.parse(raw))
    } catch {
      setError("Invalid JSON")
    }
  }

  return (
    <div className="space-y-1">
      <FieldLabel label={label} required={required} />
      <Textarea value={text} onChange={(e) => handleChange(e.target.value)} className="min-h-[80px] font-mono text-xs" />
      {error && <p className="text-xs text-red-500">{error}</p>}
    </div>
  )
}

function Section({
  title,
  defaultOpen,
  required,
  children,
}: {
  title: string
  defaultOpen?: boolean
  required?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen ?? true)
  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center justify-between rounded-lg px-1 py-2 text-sm font-semibold hover:bg-muted/50 transition-colors">
        <span>
          {title}
          {required && (
            <span className="ml-0.5 text-red-500" aria-label="required">
              *
            </span>
          )}
        </span>
        <ChevronDown className={`h-4 w-4 text-muted-foreground transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <div className="space-y-4 pb-4 pt-2">{children}</div>
      </CollapsibleContent>
    </Collapsible>
  )
}

// --- Paired array detection ---
// Matches the DD sampler pattern: an object with sibling `values` (string[])
// and `weights` (number[]) arrays of equal length. No allowlist — purely
// structural: must be exactly these two names, one all-strings, one all-numbers.

function findPairedArrays(obj: Record<string, unknown>): { key1: string; key2: string } | null {
  const values = obj.values
  const weights = obj.weights
  if (!Array.isArray(values) || !Array.isArray(weights)) return null
  if (values.length === 0 || values.length !== weights.length) return null
  if (!values.every((v) => typeof v === "string")) return null
  if (!weights.every((v) => typeof v === "number")) return null
  return { key1: "values", key2: "weights" }
}

function PairedArrayTable({
  key1,
  key2,
  arr1,
  arr2,
  onChange,
}: {
  key1: string
  key2: string
  arr1: unknown[]
  arr2: unknown[]
  onChange: (k1Vals: unknown[], k2Vals: unknown[]) => void
}) {
  function updateCell(arrIdx: 0 | 1, rowIdx: number, value: string) {
    const target = arrIdx === 0 ? [...arr1] : [...arr2]
    const original = target[rowIdx]
    target[rowIdx] = typeof original === "number" ? Number(value) : value
    if (arrIdx === 0) onChange(target, arr2)
    else onChange(arr1, target)
  }

  return (
    <div className="rounded-lg border overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="text-xs">{formatLabel(key1)}</TableHead>
            <TableHead className="text-xs w-[120px]">{formatLabel(key2)}</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {arr1.map((v1, i) => (
            <TableRow key={i}>
              <TableCell className="p-1.5">
                <Input
                  value={String(v1 ?? "")}
                  onChange={(e) => updateCell(0, i, e.target.value)}
                  className="h-8 text-xs"
                />
              </TableCell>
              <TableCell className="p-1.5">
                <Input
                  type={typeof arr2[i] === "number" ? "number" : "text"}
                  step="0.01"
                  value={String(arr2[i] ?? "")}
                  onChange={(e) => updateCell(1, i, e.target.value)}
                  className="h-8 text-xs w-full"
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

// --- Generic recursive renderer ---

function RecursiveField({
  fieldKey,
  value,
  onChange,
  meta,
  schema,
  path,
  requiredPaths,
}: {
  fieldKey: string
  value: unknown
  onChange: (v: unknown) => void
  meta?: FieldMeta
  schema?: JsonSchema | null
  path: string
  requiredPaths: Set<string>
}) {
  // Required-ness is branch-aware (same walk as the Submit-block), looked up by the
  // field's position — not the name-keyed meta — so markers can't diverge from it.
  const required = requiredPaths.has(path)
  // Data-wired pickers take precedence over the value-type dispatch below.
  if (fieldKey === "document_ids") {
    return (
      <DocumentPickerField
        label={fieldKey}
        value={value}
        onChange={onChange}
        description={meta?.description}
        required={required}
      />
    )
  }
  if (fieldKey === "parent_job_id") {
    return (
      <DatasetPickerField
        value={value}
        onChange={onChange}
        required={required}
        valueKey="job_id"
      />
    )
  }
  if (fieldKey === "data_run_id" || fieldKey === "eval_data_run_id") {
    return (
      <DatasetPickerField
        value={value}
        onChange={onChange}
        required={required}
        valueKey="run_id"
      />
    )
  }
  if (fieldKey === "model_alias") {
    return (
      <ModelAliasField
        label={fieldKey}
        value={value}
        onChange={onChange}
        description={meta?.description}
        required={required}
      />
    )
  }
  if (meta?.enumValues && typeof value === "string") {
    return (
      <EnumField
        label={fieldKey}
        value={value}
        options={meta.enumValues}
        onChange={onChange}
        description={meta.description}
        required={required}
      />
    )
  }
  if (typeof value === "boolean") {
    return <BooleanField label={fieldKey} value={value} onChange={onChange} description={meta?.description} required={required} />
  }
  if (typeof value === "number") {
    return (
      <NumberField
        label={fieldKey}
        value={value}
        onChange={(v) => onChange(v)}
        step={fieldKey.includes("rate") || fieldKey.includes("temperature") ? "0.00001" : undefined}
        description={meta?.description}
        required={required}
      />
    )
  }
  if (typeof value === "string") {
    if (value.length > 80 || fieldKey === "prompt" || fieldKey === "system_prompt" || fieldKey === "content") {
      return <LongTextField label={fieldKey} value={value} onChange={onChange} description={meta?.description} required={required} />
    }
    return <TextField label={fieldKey} value={value} onChange={onChange} description={meta?.description} required={required} />
  }
  if (Array.isArray(value)) {
    if (value.length > 0 && typeof value[0] === "object" && value[0] !== null) {
      return <ArrayOfObjectsField label={fieldKey} items={value as Record<string, unknown>[]} onChange={onChange} schema={schema} path={path} requiredPaths={requiredPaths} />
    }
    return <JsonField label={fieldKey} value={value} onChange={onChange} required={required} />
  }
  if (typeof value === "object" && value !== null) {
    return <NestedObjectField label={fieldKey} value={value as Record<string, unknown>} onChange={onChange} schema={schema} path={path} requiredPaths={requiredPaths} />
  }
  return null
}

function itemLabel(item: Record<string, unknown>, index: number, parentLabel: string): { name: string; badge: string } {
  const name =
    (item.name as string) || (item.alias as string) || (item.role as string) || `${formatLabel(parentLabel)} ${index + 1}`
  const badge =
    (item.column_type as string) || (item.processor_type as string) || (item.type as string) || ""
  return { name, badge }
}

function ArrayOfObjectsField({
  label,
  items,
  onChange,
  schema,
  path,
  requiredPaths,
}: {
  label: string
  items: Record<string, unknown>[]
  onChange: (v: unknown) => void
  schema?: JsonSchema | null
  path: string
  requiredPaths: Set<string>
}) {
  function updateItem(index: number, key: string, value: unknown) {
    const updated = items.map((item, i) => (i === index ? { ...item, [key]: value } : item))
    onChange(updated)
  }

  function updateItemBatch(index: number, updates: Record<string, unknown>) {
    const updated = items.map((item, i) => (i === index ? { ...item, ...updates } : item))
    onChange(updated)
  }

  return (
    <div className="space-y-3">
      {items.map((item, i) => {
        const { name, badge } = itemLabel(item, i, label)
        return (
          <div key={i} className="rounded-lg border p-3 space-y-3">
            <div className="flex items-center gap-2">
              <p className="text-xs font-medium text-muted-foreground">{name}</p>
              {badge && (
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{badge}</span>
              )}
            </div>
            <ObjectFields
              obj={item}
              onFieldChange={(k, v) => updateItem(i, k, v)}
              onBatchChange={(updates) => updateItemBatch(i, updates)}
              schema={schema}
              path={joinPath(path, i)}
              requiredPaths={requiredPaths}
            />
          </div>
        )
      })}
    </div>
  )
}

function NestedObjectField({
  label,
  value,
  onChange,
  schema,
  path,
  requiredPaths,
}: {
  label: string
  value: Record<string, unknown>
  onChange: (v: unknown) => void
  schema?: JsonSchema | null
  path: string
  requiredPaths: Set<string>
}) {
  return (
    <div className="space-y-2 rounded-md border-l-2 border-border/50 pl-3">
      <p className="text-xs font-medium text-muted-foreground">{formatLabel(label)}</p>
      <ObjectFields
        obj={value}
        onFieldChange={(k, v) => onChange({ ...value, [k]: v })}
        onBatchChange={(updates) => onChange({ ...value, ...updates })}
        schema={schema}
        path={path}
        requiredPaths={requiredPaths}
      />
    </div>
  )
}

function ObjectFields({
  obj,
  onFieldChange,
  onBatchChange,
  schema,
  path,
  requiredPaths,
}: {
  obj: Record<string, unknown>
  onFieldChange: (key: string, value: unknown) => void
  onBatchChange?: (updates: Record<string, unknown>) => void
  schema?: JsonSchema | null
  path: string
  requiredPaths: Set<string>
}) {
  const renderField = (key: string) => (
    <RecursiveField
      key={key}
      fieldKey={key}
      value={obj[key]}
      onChange={(v) => onFieldChange(key, v)}
      meta={getFieldMeta(key, schema)}
      schema={schema}
      path={joinPath(path, key)}
      requiredPaths={requiredPaths}
    />
  )

  // A data_designer model config: render the catalog picker for `model` (which
  // also sets the sibling `provider`) in place of the two free-text fields.
  const isModelConfig = "alias" in obj && "model" in obj && "provider" in obj
  if (isModelConfig) {
    const modelMeta = getFieldMeta("model", schema)
    const setModelProvider = (modelId: string, providerId: string) => {
      if (onBatchChange) onBatchChange({ model: modelId, provider: providerId })
      else {
        onFieldChange("model", modelId)
        onFieldChange("provider", providerId)
      }
    }
    const restKeys = Object.keys(obj).filter((k) => k !== "model" && k !== "provider")
    return (
      <>
        <ModelPickerField
          label="model"
          model={String(obj.model ?? "")}
          provider={String(obj.provider ?? "")}
          onChange={setModelProvider}
          description={modelMeta.description}
          required={requiredPaths.has(joinPath(path, "model"))}
        />
        {restKeys.map(renderField)}
      </>
    )
  }

  const paired = findPairedArrays(obj)

  if (paired) {
    const { key1, key2 } = paired
    const otherKeys = Object.keys(obj).filter((k) => k !== key1 && k !== key2)
    return (
      <>
        <PairedArrayTable
          key1={key1}
          key2={key2}
          arr1={obj[key1] as unknown[]}
          arr2={obj[key2] as unknown[]}
          onChange={(a1, a2) => {
            if (onBatchChange) {
              onBatchChange({ [key1]: a1, [key2]: a2 })
            } else {
              onFieldChange(key1, a1)
              onFieldChange(key2, a2)
            }
          }}
        />
        {otherKeys.map(renderField)}
      </>
    )
  }

  return <>{Object.keys(obj).map(renderField)}</>
}

// --- Training Form ---

const TRAINING_PARAMS_KEYS = [
  "learning_rate", "num_train_epochs", "per_device_train_batch_size",
  "max_length", "gradient_accumulation_steps",
]
const TRAINING_LORA_KEYS = [
  "use_peft", "lora_r", "lora_alpha", "lora_dropout", "load_in_4bit",
]
const TRAINING_ADVANCED_KEYS = [
  "bf16", "gradient_checkpointing", "data_path", "output_dir", "unfreeze_rank_ratio",
]
const ALL_TRAINING_KNOWN = new Set<string>([
  "algorithm", "model_name_or_path",
  ...TRAINING_PARAMS_KEYS, ...TRAINING_LORA_KEYS, ...TRAINING_ADVANCED_KEYS,
])

function hasKey(config: Record<string, unknown>, k: string) {
  return k in config && config[k] !== undefined && config[k] !== null
}

function TrainingConfigForm({
  config,
  onChange,
  schema,
}: {
  config: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
  schema?: JsonSchema | null
}) {
  const algorithm = (config.algorithm as string) || "sft"
  const isLora = algorithm.includes("lora")
  const algoMeta = getFieldMeta("algorithm", schema)

  const paramKeys = TRAINING_PARAMS_KEYS.filter((k) => hasKey(config, k))
  const loraKeys = TRAINING_LORA_KEYS.filter((k) => hasKey(config, k))
  const advancedKeys = TRAINING_ADVANCED_KEYS.filter((k) => hasKey(config, k))
  const otherKeys = Object.keys(config).filter((k) => !ALL_TRAINING_KNOWN.has(k))

  // Branch-aware required paths — same source as the Submit-block, so markers match.
  const requiredPaths = collectRequiredPaths(config, schema)
  const hasRequired = (keys: string[]) => keys.some((k) => requiredPaths.has(k))

  const tField = (key: string) => (
    <RecursiveField
      key={key}
      fieldKey={key}
      value={config[key]}
      onChange={(v) => onChange({ ...config, [key]: v })}
      meta={getFieldMeta(key, schema)}
      schema={schema}
      path={key}
      requiredPaths={requiredPaths}
    />
  )

  return (
    <div className="space-y-2">
      <Section title="Model" defaultOpen>
        {algoMeta.enumValues ? (
          <EnumField
            label="algorithm"
            value={algorithm}
            options={algoMeta.enumValues}
            onChange={(v) => onChange({ ...config, algorithm: v })}
            description={algoMeta.description}
            required={requiredPaths.has("algorithm")}
          />
        ) : (
          <TextField
            label="algorithm"
            value={algorithm}
            onChange={(v) => onChange({ ...config, algorithm: v })}
            description={algoMeta.description}
            required={requiredPaths.has("algorithm")}
          />
        )}
        <RecursiveField
          fieldKey="model_name_or_path"
          value={config.model_name_or_path ?? ""}
          onChange={(v) => onChange({ ...config, model_name_or_path: v })}
          meta={getFieldMeta("model_name_or_path", schema)}
          schema={schema}
          path="model_name_or_path"
          requiredPaths={requiredPaths}
        />
      </Section>

      {paramKeys.length > 0 && (
        <Section title="Training Parameters" defaultOpen>
          <div className="grid grid-cols-2 gap-4">{paramKeys.map(tField)}</div>
        </Section>
      )}

      {(isLora || loraKeys.length > 0) && (
        <Section title="LoRA Settings" defaultOpen>
          {loraKeys.map(tField)}
        </Section>
      )}

      {advancedKeys.length > 0 && (
        <Section key={`advanced:${hasRequired(advancedKeys)}`} title="Advanced" defaultOpen={hasRequired(advancedKeys)}>
          {advancedKeys.map(tField)}
        </Section>
      )}

      {otherKeys.length > 0 && (
        <Section key={`other:${hasRequired(otherKeys)}`} title="Other Parameters" defaultOpen={hasRequired(otherKeys)}>
          {otherKeys.map(tField)}
        </Section>
      )}
    </div>
  )
}

// --- SDG Form ---

function SdgConfigForm({
  config,
  onChange,
  schema,
  requiredKeys = [],
}: {
  config: Record<string, unknown>
  onChange: (config: Record<string, unknown>) => void
  schema?: JsonSchema | null
  requiredKeys?: string[]
}) {
  const primitiveKeys: string[] = []
  const complexKeys: string[] = []

  for (const [key, val] of Object.entries(config)) {
    if (val === null || val === undefined) continue
    if (typeof val === "object") complexKeys.push(key)
    else primitiveKeys.push(key)
  }

  // Branch-aware required paths (schema + per-recipe requiredKeys) — the single
  // source the markers, section-open, and the Submit-block all derive from.
  const requiredPaths = collectRequiredPaths(config, schema, requiredKeys)
  // Aliases declared here so a column's model_alias can be picked from them.
  const aliases = Array.isArray(config.model_configs)
    ? (config.model_configs as Record<string, unknown>[]).map((c) => String(c.alias ?? ""))
    : []

  return (
    <SdgModelAliasesContext.Provider value={aliases}>
      <div className="space-y-2">
        {primitiveKeys.length > 0 && (
          <Section title="Generation Settings" defaultOpen>
            <div className="grid grid-cols-2 gap-4">
              {primitiveKeys.map((key) => (
                <RecursiveField
                  key={key}
                  fieldKey={key}
                  value={config[key]}
                  onChange={(v) => onChange({ ...config, [key]: v })}
                  meta={getFieldMeta(key, schema)}
                  schema={schema}
                  path={key}
                  requiredPaths={requiredPaths}
                />
              ))}
            </div>
          </Section>
        )}

        {complexKeys.map((key) => {
          const value = config[key]
          const isRequired = requiredPaths.has(key)
          // Skip empty arrays, except document_ids (a picker, usable when empty) and
          // required keys (hiding them would block Submit with no way to re-add).
          if (
            Array.isArray(value) &&
            value.length === 0 &&
            key !== "document_ids" &&
            !isRequired
          ) {
            return null
          }
          const open = isRequired || key === "document_ids"
          // Key on `open` so the section remounts (re-seeding its open state) when
          // `open` flips false→true as the schema loads async after mount.
          return (
            <Section key={`${key}:${open}`} title={formatLabel(key)} defaultOpen={open} required={isRequired}>
              <RecursiveField
                fieldKey={key}
                value={value}
                onChange={(v) => onChange({ ...config, [key]: v })}
                meta={getFieldMeta(key, schema)}
                schema={schema}
                path={key}
                requiredPaths={requiredPaths}
              />
            </Section>
          )
        })}
      </div>
    </SdgModelAliasesContext.Provider>
  )
}

export function RecipeConfigForm({ type, config, onChange, schema, requiredKeys }: RecipeConfigFormProps) {
  const typeSchema = schema ? (schema[type] as JsonSchema | undefined) : null
  if (type === "training") {
    return <TrainingConfigForm config={config} onChange={onChange} schema={typeSchema} />
  }
  return (
    <SdgConfigForm
      config={config}
      onChange={onChange}
      schema={typeSchema}
      requiredKeys={requiredKeys}
    />
  )
}
