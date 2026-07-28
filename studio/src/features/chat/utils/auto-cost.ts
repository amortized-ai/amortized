import type { ChatMessage, ToolResult } from "../types"
import type { MlflowGatewayRoute } from "@/types/api"
import { friendlyModelLabel } from "./workflow-options"

const SAMPLE_COUNT_RE = /\b(\d+)\s*samples?\b/i
const MODEL_ID_RE = /\b(openai\/gpt-[\w-]+|anthropic\/claude-[\w-]+|vertex_ai\/claude-[\w-]+)\b/i

const CONFIRMATION_SIGNALS = [
  /here'?s (?:your )?(?:\w+ )?plan/i,
  /ready to (?:go|run|generate|start|submit)/i,
  /looks good,? run it/i,
  /start generating/i,
  /let'?s (?:go|run|do it)/i,
  /\bconfirm/i,
]

const TRAINING_SIGNALS = [
  /fine-?tun/i,
  /\bLoRA\b/,
  /\bQLoRA\b/,
  /\bSFT\b/,
  /training method/i,
  /student model/i,
]

const TEACHER_MODEL_SIGNALS = [
  /which teacher model/i,
  /pick (?:a|your) teacher model/i,
  /choose (?:a|your) teacher model/i,
  /(?:one|only) teacher model.*available/i,
  /which model.*(?:generate|create).*(?:data|samples)\?/i,
]

const COMPARE_SDG_TOOLS = new Set(["compare sdg models", "compare_sdg_models"])

type Phase = "sdg" | "training"

function phaseTagToPhase(tag: string): Phase | null {
  const part = tag.split(":")[0]
  if (part === "sdg" || part === "training") return part
  return null
}

function detectPhase(messages: ChatMessage[]): Phase {
  const allText = messages.map((m) => m.content).join("\n")
  if (TRAINING_SIGNALS.some((re) => re.test(allText))) return "training"
  return "sdg"
}

function isTeacherModelStep(content: string): boolean {
  return TEACHER_MODEL_SIGNALS.some((re) => re.test(content))
}

const STUDENT_MODEL_SIGNALS = [
  /which student model/i,
  /which (?:student )?model.*(?:train|fine)/i,
  /model size.*(?:train|fine)/i,
  /model.*would you like to train/i,
  /Qwen.*LoRA|Llama.*LoRA/i,
  /Qwen3-0\.6B.*Qwen3-4B|Qwen3.*fastest.*largest/i,
]

function isStudentModelStep(content: string): boolean {
  return STUDENT_MODEL_SIGNALS.some((re) => re.test(content))
}

const TRAINING_METHOD_SIGNALS = [
  /which training method/i,
  /training method/i,
  /LoRA SFT.*QLoRA|QLoRA.*Full SFT/i,
]

function isTrainingMethodStep(content: string): boolean {
  return TRAINING_METHOD_SIGNALS.some((re) => re.test(content))
}

function isConfirmationStep(content: string): boolean {
  return CONFIRMATION_SIGNALS.some((re) => re.test(content))
}

function extractSampleCount(messages: ChatMessage[], latestContent?: string): number {
  if (latestContent) {
    const match = SAMPLE_COUNT_RE.exec(latestContent)
    if (match) return parseInt(match[1]!, 10)
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!
    const match = SAMPLE_COUNT_RE.exec(msg.content)
    if (match) return parseInt(match[1]!, 10)
    if (msg.selectedOptionValue) {
      const optMatch = SAMPLE_COUNT_RE.exec(msg.selectedOptionValue)
      if (optMatch) return parseInt(optMatch[1]!, 10)
    }
    for (const tool of msg.toolResults) {
      const toolMatch = SAMPLE_COUNT_RE.exec(tool.result)
      if (toolMatch) return parseInt(toolMatch[1]!, 10)
    }
  }
  return 50
}

const MODEL_LABEL_MAP: Record<string, string> = {
  "gpt-4o-mini": "openai/gpt-4o-mini",
  "gpt-4o mini": "openai/gpt-4o-mini",
  "claude haiku": "vertex_ai/claude-haiku-4-5@20251001",
  "claude sonnet": "vertex_ai/claude-sonnet-4@20250514",
  "gpt-4o": "openai/gpt-4o",
}

function extractModel(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const text = messages[i]!.content
    const match = MODEL_ID_RE.exec(text)
    if (match) return match[1]!
    const lower = text.toLowerCase()
    const sortedEntries = Object.entries(MODEL_LABEL_MAP).sort((a, b) => b[0].length - a[0].length)
    for (const [label, id] of sortedEntries) {
      if (lower.includes(label)) return id
    }
  }
  return "openai/gpt-4o-mini"
}

function extractStudentModelId(messages: ChatMessage[]): string {
  const modelMap: Record<string, string> = {
    "qwen3-0.6b": "qwen3-0.6b",
    "qwen/qwen3-0.6b": "qwen3-0.6b",
    "qwen2.5-1.5b": "qwen2.5-1.5b",
    "qwen/qwen2.5-1.5b-instruct": "qwen2.5-1.5b",
    "qwen3-4b": "qwen3-4b",
    "qwen/qwen3-4b": "qwen3-4b",
    "llama-3.1-8b": "llama-3.1-8b",
    "meta-llama/llama-3.1-8b": "llama-3.1-8b",
  }
  for (let i = messages.length - 1; i >= 0; i--) {
    const lower = messages[i]!.content.toLowerCase()
    for (const [pattern, id] of Object.entries(modelMap)) {
      if (lower.includes(pattern)) return id
    }
  }
  return "qwen3-0.6b"
}

function hasCostToolBeenCalled(
  messages: ChatMessage[],
  toolSet: Set<string>,
): boolean {
  return messages.some((m) =>
    m.toolResults.some((t) => {
      const normalized = t.name.replace(/_/g, " ").toLowerCase()
      return toolSet.has(normalized) || toolSet.has(t.name)
    }),
  )
}

function getBaseUrl(): string {
  return import.meta.env.VITE_API_URL ?? ""
}

async function callCostApi(
  path: string,
  body: Record<string, unknown>,
  toolName: string,
): Promise<ToolResult | null> {
  try {
    const resp = await fetch(`${getBaseUrl()}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
    if (!resp.ok) return null
    const data = await resp.json()
    return {
      name: toolName,
      result: JSON.stringify(data),
      collapsed: true,
    }
  } catch {
    return null
  }
}

export async function autoCostEstimate(
  messages: ChatMessage[],
  latestContent: string,
  phaseTag?: string | null,
  routes?: MlflowGatewayRoute[],
): Promise<ToolResult | null> {
  // Use the phase tag from the agent when available — it's authoritative.
  // Only fall back to regex detection when no tag is present.
  const phase = phaseTag ? phaseTagToPhase(phaseTag) : detectPhase(messages)
  if (!phase) return null

  const numSamples = extractSampleCount(messages, latestContent)

  // Check confirmation FIRST — "Teacher Model" in the table was falsely
  // triggering the teacher model comparison instead of the cost summary.
  // Skip if the agent already shows a cost inline (avoids conflicting values).
  if (isConfirmationStep(latestContent)) {
    if (/\$\d+(?:\.\d+)?/.test(latestContent)) return null

    const confirmPhase = phaseTag ? phaseTagToPhase(phaseTag)
      : /\btraining\s+plan\b|\bstudent model\b|\bfine-?tun.*plan\b/i.test(latestContent) ? "training"
      : phase

    if (confirmPhase === "sdg") {
      if (hasCostToolBeenCalled(messages, new Set(["estimate sdg cost", "estimate_sdg_cost"]))) return null
      const model = extractModel(messages)
      return callCostApi("/api/v1/costs/sdg", { num_samples: numSamples, model }, "estimate sdg cost")
    }
    if (confirmPhase === "training") {
      if (hasCostToolBeenCalled(messages, new Set(["training cost summary"]))) return null
      const modelId = extractStudentModelId(messages)
      return callCostApi("/api/v1/costs/training/method", { model_id: modelId, num_samples: numSamples }, "training cost summary")
    }
    return null
  }

  // Model/method selection steps (only if NOT a confirmation)
  if (phase === "sdg" && isTeacherModelStep(latestContent)) {
    if (hasCostToolBeenCalled(messages, COMPARE_SDG_TOOLS)) return null
    const body: Record<string, unknown> = { num_samples: numSamples }
    if (routes && routes.length > 0) {
      body.models = routes.map((r) => ({
        model_id: `${r.model.provider}/${r.model.name}`,
        label: friendlyModelLabel(r),
        description: `${r.model.provider} · ${r.model.name}`,
      }))
    }
    return callCostApi("/api/v1/costs/sdg/compare", body, "compare sdg models")
  }

  if (phase === "training" && isStudentModelStep(latestContent)) {
    if (hasCostToolBeenCalled(messages, new Set(["estimate training cost", "estimate_training_cost"]))) return null
    return callCostApi("/api/v1/costs/training", { num_samples: numSamples }, "estimate training cost")
  }

  if (phase === "training" && isTrainingMethodStep(latestContent)) {
    if (hasCostToolBeenCalled(messages, new Set(["estimate training method cost", "estimate_training_method_cost"]))) return null
    const modelId = extractStudentModelId(messages)
    return callCostApi("/api/v1/costs/training/method", { model_id: modelId, num_samples: numSamples }, "estimate training method cost")
  }

  return null
}
