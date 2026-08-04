import type { ChatMessage, PlanStep, PlanPhase, PhasePlan } from "../types"
import { getLatestPhase, parsePhaseTag } from "./workflow-options"

const PHASE_LABELS: Record<PlanPhase, string> = {
  sdg: "Data Generation",
  training: "Model Training",
  eval: "Evaluation",
}

function isValidPhase(s: string): s is PlanPhase {
  return s === "sdg" || s === "training" || s === "eval"
}

function tryParseToolResult(result: string): Record<string, unknown> | null {
  try {
    const parsed = typeof result === "string" ? JSON.parse(result) : result
    return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : null
  } catch {
    return null
  }
}

const TOOL_LABELS: Record<string, string> = {
  "list_models": "Checking available models",
  "get_config": "Checking platform readiness",
  "get_model_pricing": "Comparing model pricing",
  "show_model_pricing": "Comparing model pricing",
  "compare_sdg_models": "Comparing model pricing",
  "estimate_sdg_cost": "Estimating generation cost",
  "estimate_training_cost": "Estimating training cost",
  "estimate_training_resources": "Estimating training resources",
  "estimate_training_method_cost": "Comparing training methods",
  "show_vram_estimate": "Estimating GPU requirements",
  "create_job": "Submitting job",
  "get_job_detail": "Checking job status",
  "get_job_logs": "Reading job logs",
}

const PHASE_STEP_LABELS: Record<string, string> = {
  "understand_task": "Understanding your task",
  "load_skill": "Loading skill guide",
  "gather_requirements": "Gathering requirements",
  "confirm": "Reviewing configuration",
  "review": "Checking results",
}

export function deriveDynamicPlan(messages: ChatMessage[]): PhasePlan | null {
  let latestPhase: PlanPhase | null = null
  const seenLabels = new Set<string>()
  const steps: Array<{ label: string }> = []

  for (const msg of messages) {
    for (const tool of msg.toolResults) {
      if (tool.name === "signal_phase" || tool.name === "signal phase") {
        const data = tryParseToolResult(tool.result)
        if (!data) continue
        const phase = String(data.phase ?? "")
        if (!isValidPhase(phase)) continue
        const step = String(data.step ?? "")

        if (latestPhase && phase !== latestPhase) {
          seenLabels.clear()
          steps.length = 0
        }
        latestPhase = phase

        const label = PHASE_STEP_LABELS[step]
        if (label && !seenLabels.has(label)) {
          seenLabels.add(label)
          steps.push({ label })
        }
        continue
      }

      const label = TOOL_LABELS[tool.name]
      if (label && !seenLabels.has(label)) {
        seenLabels.add(label)
        steps.push({ label })
      }
    }
  }

  if (!latestPhase || steps.length === 0) return null

  return {
    phase: latestPhase,
    label: PHASE_LABELS[latestPhase],
    steps: steps.map((s, i): PlanStep => ({
      label: s.label,
      status: i < steps.length - 1 ? "completed" : "active",
    })),
  }
}

interface StepDef {
  label: string
  matchSteps: string[]
}

const SDG_STEPS: StepDef[] = [
  { label: "Understanding your task", matchSteps: ["understand_task"] },
  { label: "Loading skill guide", matchSteps: ["load_skill"] },
  { label: "Gathering requirements", matchSteps: ["gather_requirements"] },
  { label: "Checking models & estimating cost", matchSteps: ["estimate_cost"] },
  { label: "Reviewing configuration", matchSteps: ["confirm"] },
  { label: "Submitting job", matchSteps: ["execute"] },
  { label: "Checking results", matchSteps: ["review"] },
]

const TRAINING_STEPS: StepDef[] = [
  { label: "Understanding your task", matchSteps: ["understand_task"] },
  { label: "Loading training guide", matchSteps: ["load_skill"] },
  { label: "Gathering requirements", matchSteps: ["gather_requirements"] },
  { label: "Selecting model & estimating cost", matchSteps: ["estimate_cost"] },
  { label: "Reviewing configuration", matchSteps: ["confirm"] },
  { label: "Submitting training job", matchSteps: ["execute"] },
  { label: "Checking results", matchSteps: ["review"] },
]

const STATIC_PHASE_CONFIG: Partial<Record<PlanPhase, { label: string; steps: StepDef[] }>> = {
  sdg: { label: "Data Generation", steps: SDG_STEPS },
  training: { label: "Model Training", steps: TRAINING_STEPS },
}

function resolveStepIndex(stepDefs: StepDef[], currentStep: string): number {
  for (let i = stepDefs.length - 1; i >= 0; i--) {
    if (stepDefs[i]!.matchSteps.includes(currentStep)) return i
  }
  return 0
}

export function derivePhasePlan(messages: ChatMessage[]): PhasePlan | null {
  const latestPhase = getLatestPhase(messages)
  if (!latestPhase) return null

  const parsed = parsePhaseTag(latestPhase)
  if (!parsed) return null

  const config = STATIC_PHASE_CONFIG[parsed.phase]
  if (!config) return null
  const activeIdx = resolveStepIndex(config.steps, parsed.step)
  const isCompleted = parsed.step === "review"

  const planSteps: PlanStep[] = config.steps.map((def, i): PlanStep => {
    if (isCompleted) return { label: def.label, status: "completed" }
    if (i < activeIdx) return { label: def.label, status: "completed" }
    if (i === activeIdx) return { label: def.label, status: "active" }
    return { label: def.label, status: "pending" }
  })

  return { phase: parsed.phase, label: config.label, steps: planSteps }
}

export function derivePlan(messages: ChatMessage[]): PhasePlan | null {
  return deriveDynamicPlan(messages) ?? derivePhasePlan(messages)
}
