import { useState } from "react"
import {
  Card,
  CardContent,
  CardFooter,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Loader2, ChevronDown, ChevronRight, Rocket } from "lucide-react"
import type { ProposedAction } from "../types"

function extractSdgSummary(config: Record<string, unknown>): [string, string][] {
  const rows: [string, string][] = []

  const columns = config.columns as Array<Record<string, unknown>> | undefined
  if (columns?.length) {
    const types = columns.map((c) => c.column_type || c.name).join(", ")
    rows.push(["Columns", `${columns.length} (${types})`])
  }

  const modelConfigs = config.model_configs as Array<Record<string, unknown>> | undefined
  if (modelConfigs?.length) {
    const model = modelConfigs[0]?.model ?? modelConfigs[0]?.alias ?? "unknown"
    rows.push(["Teacher model", String(model)])
  }

  if (config.num_records) rows.push(["Samples", String(config.num_records)])
  if (config.mode) rows.push(["Mode", String(config.mode)])
  if (config.topic) rows.push(["Topic", String(config.topic)])

  const docIds = config.document_ids as string[] | undefined
  if (docIds?.length) rows.push(["Documents", `${docIds.length} document${docIds.length > 1 ? "s" : ""}`])

  return rows
}

function extractTrainingSummary(config: Record<string, unknown>): [string, string][] {
  const rows: [string, string][] = []

  if (config.algorithm) rows.push(["Algorithm", String(config.algorithm).toUpperCase()])
  if (config.model_name_or_path) rows.push(["Model", String(config.model_name_or_path)])
  if (config.num_train_epochs) rows.push(["Epochs", String(config.num_train_epochs)])
  if (config.learning_rate) rows.push(["Learning rate", String(config.learning_rate)])
  if (config.effective_batch_size ?? config.per_device_train_batch_size)
    rows.push(["Batch size", String(config.effective_batch_size ?? config.per_device_train_batch_size)])
  if (config.max_length) rows.push(["Max seq length", String(config.max_length)])
  if (config.unfreeze_rank_ratio) rows.push(["Unfreeze ratio", String(config.unfreeze_rank_ratio)])
  if (config.nproc_per_node) rows.push(["GPUs", String(config.nproc_per_node)])

  return rows
}

function extractEvalSummary(config: Record<string, unknown>): [string, string][] {
  const rows: [string, string][] = []

  // Current configs use `endpoint` (single model under evaluation);
  // endpoint_base/endpoint_tuned are the legacy two-model form.
  const endpointModel = (config.endpoint as Record<string, unknown> | undefined)?.model
  const baseModel = (config.endpoint_base as Record<string, unknown> | undefined)?.model
  const tunedModel = (config.endpoint_tuned as Record<string, unknown> | undefined)?.model
  if (endpointModel) rows.push(["Model", String(endpointModel)])
  if (baseModel) rows.push(["Base model", String(baseModel)])
  if (tunedModel) rows.push(["Tuned model", String(tunedModel)])
  if (config.training_job_id) rows.push(["Training job", String(config.training_job_id).slice(0, 8)])
  if (config.model_name_or_path && !endpointModel) rows.push(["Model", String(config.model_name_or_path)])
  if (config.eval_data_run_id) rows.push(["Eval data run", String(config.eval_data_run_id).slice(0, 8)])
  const rubric = config.rubric as { name?: string }[] | undefined
  if (rubric?.length) rows.push(["Rubric", rubric.map((c) => c.name).join(", ")])
  if (config.max_samples) rows.push(["Max samples", String(config.max_samples)])

  return rows
}

function extractServeSummary(config: Record<string, unknown>): [string, string][] {
  const rows: [string, string][] = []

  if (config.training_job_id)
    rows.push(["Training job", String(config.training_job_id).slice(0, 8)])
  if (config.model_name_or_path) rows.push(["Model", String(config.model_name_or_path)])
  if (config.served_model_name) rows.push(["Served name", String(config.served_model_name)])
  if (config.nproc_per_node) rows.push(["GPUs", String(config.nproc_per_node)])
  if (config.max_model_len) rows.push(["Max model len", String(config.max_model_len)])

  return rows
}

function extractConfigSummary(jobType: string | undefined, config: Record<string, unknown>): [string, string][] {
  if (jobType === "sdg") return extractSdgSummary(config)
  if (jobType === "training") return extractTrainingSummary(config)
  if (jobType === "eval") return extractEvalSummary(config)
  if (jobType === "serve") return extractServeSummary(config)

  const rows: [string, string][] = []
  for (const [key, value] of Object.entries(config)) {
    if (typeof value !== "object") rows.push([key, String(value)])
  }
  return rows.slice(0, 8)
}

interface ActionCardProps {
  action: ProposedAction
  onConfirm: () => void
  onReject: () => void
}

export function ActionCard({ action, onConfirm, onReject }: ActionCardProps) {
  const [submitting, setSubmitting] = useState(false)
  const [showRaw, setShowRaw] = useState(false)

  const handleConfirm = async () => {
    setSubmitting(true)
    try {
      await onConfirm()
    } catch {
      setSubmitting(false)
    }
  }

  const summary = action.config ? extractConfigSummary(action.jobType, action.config) : []

  return (
    <Card className="border-primary/30 bg-primary/5 overflow-hidden">
      <CardContent className="p-4 pb-3">
        <div className="flex items-center gap-2 mb-3">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary/10">
            <Rocket className="h-3.5 w-3.5 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium">{action.action}</p>
            <p className="text-xs text-muted-foreground">{action.description}</p>
          </div>
        </div>

        {summary.length > 0 && (
          <div className="rounded-lg border border-border/50 overflow-hidden">
            <table className="w-full text-xs">
              <tbody>
                {summary.map(([label, value], i) => (
                  <tr key={label} className={i % 2 === 0 ? "bg-muted/30" : "bg-transparent"}>
                    <td className="px-3 py-1.5 text-muted-foreground font-medium w-[140px]">{label}</td>
                    <td className="px-3 py-1.5 break-all">{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {action.config && (
          <button
            type="button"
            className="flex items-center gap-1 mt-2 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
            onClick={() => setShowRaw(!showRaw)}
          >
            {showRaw ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            Full config
          </button>
        )}

        {showRaw && action.config && (
          <pre className="mt-1 rounded-md bg-muted/50 p-2 text-[11px] font-mono max-h-48 overflow-auto whitespace-pre-wrap break-all">
            {JSON.stringify(action.config, null, 2)}
          </pre>
        )}
      </CardContent>

      <CardFooter className="gap-2 px-4 pb-4 pt-0">
        <Button size="sm" onClick={handleConfirm} disabled={submitting}>
          {submitting ? (
            <>
              <Loader2 className="mr-1 h-3 w-3 animate-spin" />
              Submitting
            </>
          ) : (
            "Confirm"
          )}
        </Button>
        <Button size="sm" variant="outline" onClick={onReject} disabled={submitting}>
          Cancel
        </Button>
      </CardFooter>
    </Card>
  )
}
