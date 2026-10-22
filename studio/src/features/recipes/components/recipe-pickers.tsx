import { useContext, useEffect, useRef, useState } from "react"
import { Check, ChevronDown, X } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { cn } from "@/lib/utils"
import { useDocuments } from "@/features/documents/api/use-documents"
import { useDatasets } from "@/features/datasets/api/use-datasets"
import { useGatewayModels } from "../api/use-gateway-models"
import { FieldLabel, FieldDescription } from "./recipe-fields"
import { SdgModelAliasesContext } from "./recipe-field-utils"

// Separator used to encode provider+model into a single Radix Select value
// (model ids can contain ":" / "-", so use a control char that never appears).
const MODEL_SEP = "\u0001"

interface PickerFieldProps {
  label: string
  value: unknown
  onChange: (v: unknown) => void
  description?: string
  required?: boolean
}

// --- document_ids → multi-select document picker (value = string[]) ---

export function DocumentPickerField({
  label,
  value,
  onChange,
  description,
  required,
}: PickerFieldProps) {
  const [open, setOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const { data: documents = [] } = useDocuments()
  const selected = Array.isArray(value)
    ? (value.filter((v): v is string => typeof v === "string"))
    : []
  const nameFor = (id: string) =>
    documents.find((d) => d.document_id === id)?.filename ?? id

  function toggle(id: string) {
    onChange(
      selected.includes(id)
        ? selected.filter((s) => s !== id)
        : [...selected, id],
    )
  }

  // Collapse when the user clicks outside the field (combobox behavior — the
  // Collapsible alone only closes when its trigger is clicked again).
  useEffect(() => {
    if (!open) return
    function onPointerDown(e: PointerEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener("pointerdown", onPointerDown)
    return () => document.removeEventListener("pointerdown", onPointerDown)
  }, [open])

  return (
    <div
      ref={containerRef}
      className="space-y-1.5"
      onKeyDown={(e) => {
        if (e.key === "Escape") setOpen(false)
      }}
    >
      <FieldLabel label={label} required={required} />
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((id) => (
            <span
              key={id}
              className="inline-flex items-center gap-1 rounded bg-muted px-1.5 py-0.5 text-xs"
            >
              <span className="max-w-[160px] truncate">{nameFor(id)}</span>
              <button
                type="button"
                onClick={() => toggle(id)}
                className="text-muted-foreground hover:text-foreground"
                aria-label={`Remove ${nameFor(id)}`}
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex h-9 w-full items-center justify-between rounded-md border border-input bg-transparent px-3 text-sm shadow-xs transition-colors hover:border-border">
          <span className="text-muted-foreground">
            {selected.length > 0 ? `${selected.length} selected` : "Select documents..."}
          </span>
          <ChevronDown
            className={cn("h-4 w-4 opacity-50 transition-transform", open && "rotate-180")}
          />
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="mt-1 rounded-md border">
            <Command>
              <CommandInput placeholder="Search documents..." className="h-9" />
              <CommandList className="max-h-[180px]">
                <CommandEmpty>No documents found.</CommandEmpty>
                {documents.map((doc) => {
                  const isSel = selected.includes(doc.document_id)
                  return (
                    <CommandItem
                      key={doc.document_id}
                      value={`${doc.filename} ${doc.document_id}`}
                      onSelect={() => {
                        toggle(doc.document_id)
                        setOpen(false)
                      }}
                    >
                      <Check
                        className={cn("h-4 w-4 shrink-0", isSel ? "opacity-100" : "opacity-0")}
                      />
                      <span className="min-w-0 flex-1 truncate">{doc.filename}</span>
                    </CommandItem>
                  )
                })}
              </CommandList>
            </Command>
          </div>
        </CollapsibleContent>
      </Collapsible>
      <FieldDescription text={description} />
    </div>
  )
}

// --- dataset picker ---
// Drives a dataset reference field. `valueKey` selects what the chosen dataset
// contributes: its amortized job id (parent_job_id) or its MLflow run id
// (data_run_id / eval_data_run_id, which also reach job-less uploads & splits).

function datasetSubtitle(topic: string, samples: string): string {
  const n = Number(samples)
  const samplesLabel = samples && Number.isFinite(n) ? `${n.toLocaleString()} samples` : ""
  return [topic, samplesLabel].filter(Boolean).join(" · ")
}

export function DatasetPickerField({
  value,
  onChange,
  description,
  required,
  valueKey = "job_id",
}: Omit<PickerFieldProps, "label"> & { valueKey?: "job_id" | "run_id" }) {
  const { data: datasets = [] } = useDatasets()
  const options = datasets
    .map((d) => ({
      key: valueKey === "run_id" ? d.run_id : (d.tags.job_id ?? ""),
      name: d.name,
      subtitle: datasetSubtitle(d.tags.dataset_topic ?? "", d.tags.num_samples ?? ""),
    }))
    .filter((o) => o.key)
  const current = typeof value === "string" ? value : ""
  const selected = options.find((o) => o.key === current)

  return (
    <div className="space-y-1">
      {/* Always labelled "Dataset" — the user is picking a dataset, not the raw
          parent_job_id / data_run_id field it's stored in. */}
      <FieldLabel label="Dataset" required={required} />
      <Select value={current || undefined} onValueChange={(v) => onChange(v)}>
        <SelectTrigger className="h-9">
          {/* Custom trigger content so it stays single-line (the items are two-line). */}
          {selected ? (
            <span className="truncate">{selected.name}</span>
          ) : current ? (
            <span className="truncate text-muted-foreground">Unknown dataset ({current.slice(0, 8)})</span>
          ) : (
            <span className="text-muted-foreground">Select a dataset...</span>
          )}
        </SelectTrigger>
        <SelectContent>
          {current && !selected && (
            <SelectItem value={current} disabled className="text-muted-foreground">
              Unknown dataset ({current.slice(0, 8)})
            </SelectItem>
          )}
          {options.map((o) => (
            <SelectItem key={o.key} value={o.key} textValue={o.name}>
              <div className="flex flex-col">
                <span>{o.name}</span>
                {o.subtitle && (
                  <span className="text-xs text-muted-foreground">{o.subtitle}</span>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription text={description ?? "The dataset to use (produced by a previous SDG job)."} />
    </div>
  )
}

// --- model_configs[].model → catalog dropdown (sets model + provider) ---

export function ModelPickerField({
  label,
  model,
  provider,
  onChange,
  description,
  required,
}: {
  label: string
  model: string
  provider: string
  onChange: (modelId: string, providerId: string) => void
  description?: string
  required?: boolean
}) {
  const { data } = useGatewayModels()
  const models = data?.models ?? []
  const known = models.some((m) => m.name === model && m.provider === provider)
  const current = model ? `${provider}${MODEL_SEP}${model}` : ""

  // Group the catalog by provider (e.g. "gateway", "openai") for the dropdown.
  const groups = new Map<string, typeof models>()
  for (const m of models) {
    const arr = groups.get(m.provider) ?? []
    arr.push(m)
    groups.set(m.provider, arr)
  }

  return (
    <div className="space-y-1">
      <FieldLabel label={label} required={required} />
      <Select
        value={current || undefined}
        onValueChange={(v) => {
          const idx = v.indexOf(MODEL_SEP)
          onChange(v.slice(idx + 1), v.slice(0, idx))
        }}
      >
        <SelectTrigger className="h-9">
          <SelectValue placeholder="Select a model..." />
        </SelectTrigger>
        <SelectContent>
          {model && !known && (
            <SelectItem value={current} disabled className="text-muted-foreground">
              {model} (unavailable)
            </SelectItem>
          )}
          {[...groups.entries()].map(([prov, mods]) => (
            <SelectGroup key={prov}>
              <SelectLabel className="text-xs text-muted-foreground">{prov}</SelectLabel>
              {mods.map((m) => (
                <SelectItem
                  key={`${m.provider}${MODEL_SEP}${m.name}`}
                  value={`${m.provider}${MODEL_SEP}${m.name}`}
                >
                  {m.name}
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
      {/* Read-only: the provider is derived from the selected model (not separately
          editable), but surfaced so the user knows which provider will be used. */}
      {provider && (
        <p className="text-xs text-muted-foreground">
          Provider: <span className="font-medium text-foreground">{provider}</span>
        </p>
      )}
      <FieldDescription text={description} />
    </div>
  )
}

// --- column model_alias → dropdown of aliases from this config's model_configs ---

export function ModelAliasField({
  label,
  value,
  onChange,
  description,
  required,
}: PickerFieldProps) {
  const aliases = useContext(SdgModelAliasesContext).filter(Boolean)
  const current = typeof value === "string" ? value : ""
  const known = aliases.includes(current)

  return (
    <div className="space-y-1">
      <FieldLabel label={label} required={required} />
      <Select value={current || undefined} onValueChange={(v) => onChange(v)}>
        <SelectTrigger className="h-9">
          <SelectValue placeholder="Select a model alias..." />
        </SelectTrigger>
        <SelectContent>
          {current && !known && (
            <SelectItem value={current} disabled className="text-muted-foreground">
              {current} (undefined alias)
            </SelectItem>
          )}
          {aliases.map((a) => (
            <SelectItem key={a} value={a}>
              {a}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <FieldDescription text={description} />
    </div>
  )
}
