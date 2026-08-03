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

export function deriveDynamicPlan(messages: ChatMessage[]): PhasePlan | null {
  let latestPhase: PlanPhase | null = null
  let highestStepIdx = -1

  for (const msg of messages) {
    for (const tool of msg.toolResults) {
      if (tool.name !== "signal_phase" && tool.name !== "signal phase") continue
      const data = tryParseToolResult(tool.result)
      if (!data) continue
      const phase = String(data.phase ?? "")
      if (!isValidPhase(phase)) continue
      const step = String(data.step ?? "")

      if (latestPhase && phase !== latestPhase) {
        highestStepIdx = -1
      }
      latestPhase = phase

      const config = STATIC_PHASE_CONFIG[phase]
      if (!config) continue
      const idx = resolveStepIndex(config.steps, step)
      if (idx > highestStepIdx) highestStepIdx = idx
    }
  }

  if (!latestPhase || highestStepIdx < 0) return null

  const config = STATIC_PHASE_CONFIG[latestPhase]
  if (!config) return null

  const steps: PlanStep[] = config.steps.slice(0, highestStepIdx + 1).map((def, i) => ({
    label: def.label,
    status: i < highestStepIdx ? "completed" as const : "active" as const,
  }))

  return { phase: latestPhase, label: PHASE_LABELS[latestPhase], steps }
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
