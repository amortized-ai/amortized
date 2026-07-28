import type { ChatMessage, PlanStep, PlanPhase, PhasePlan, ProgressStep } from "../types"
import { getLatestPhase, parsePhaseTag } from "./workflow-options"

interface StepDef {
  label: string
  matchSteps: string[]
}

const SDG_STEPS: StepDef[] = [
  { label: "Understand task", matchSteps: ["understand_task", "load_skill"] },
  { label: "Configure", matchSteps: ["gather_requirements"] },
  { label: "Estimate cost", matchSteps: ["estimate_cost", "confirm"] },
  { label: "Generate data", matchSteps: ["execute", "review"] },
]

const TRAINING_STEPS: StepDef[] = [
  { label: "Select model", matchSteps: ["understand_task", "load_skill"] },
  { label: "Configure", matchSteps: ["gather_requirements"] },
  { label: "Estimate cost", matchSteps: ["estimate_cost", "confirm"] },
  { label: "Train model", matchSteps: ["execute", "review"] },
]

const PHASE_CONFIG: Record<"sdg" | "training", { label: string; steps: StepDef[] }> = {
  sdg: { label: "Data Generation", steps: SDG_STEPS },
  training: { label: "Model Training", steps: TRAINING_STEPS },
}

const PHASE_LABELS: Record<PlanPhase, string> = {
  sdg: "Data Generation",
  training: "Model Training",
  eval: "Evaluation",
}

function resolveStepIndex(steps: StepDef[], currentStep: string): number {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i]!.matchSteps.includes(currentStep)) return i
  }
  return 0
}

export function derivePhasePlan(messages: ChatMessage[]): PhasePlan | null {
  const latestPhase = getLatestPhase(messages)
  if (!latestPhase) return null

  const parsed = parsePhaseTag(latestPhase)
  if (!parsed) return null

  const config = PHASE_CONFIG[parsed.phase]
  if (!config) return null
  const activeIdx = resolveStepIndex(config.steps, parsed.step)
  const isCompleted = parsed.step === "review"

  const steps: PlanStep[] = config.steps.map((def, i): PlanStep => {
    if (isCompleted) return { label: def.label, status: "completed" }
    if (i < activeIdx) return { label: def.label, status: "completed" }
    if (i === activeIdx) return { label: def.label, status: "active" }
    return { label: def.label, status: "pending" }
  })

  return { phase: parsed.phase, label: config.label, steps }
}

function isValidPhase(p: string): p is PlanPhase {
  return p === "sdg" || p === "training" || p === "eval"
}

export function extractProgressSteps(messages: ChatMessage[]): ProgressStep[] {
  const steps: ProgressStep[] = []
  const seen = new Map<string, number>()

  for (const msg of messages) {
    for (const tool of msg.toolResults ?? []) {
      if (tool.name !== "signal progress" && tool.name !== "signal_progress") continue
      try {
        const data = typeof tool.result === "string" ? JSON.parse(tool.result) : tool.result
        const { phase, step_id, label, status } = data as Record<string, string | undefined>
        if (!step_id || !label || !phase) continue
        if (!isValidPhase(phase)) continue

        const existing = seen.get(step_id)
        if (existing !== undefined) {
          steps[existing] = { phase, stepId: step_id, label, status: status === "completed" ? "completed" : "active" }
        } else {
          seen.set(step_id, steps.length)
          steps.push({ phase, stepId: step_id, label, status: status === "completed" ? "completed" : "active" })
        }
      } catch { /* ignore malformed */ }
    }
  }
  return steps
}

export function deriveDynamicPlan(messages: ChatMessage[]): PhasePlan | null {
  const progressSteps = extractProgressSteps(messages)
  if (progressSteps.length === 0) return null

  const lastStep = progressSteps[progressSteps.length - 1]!
  const phase = lastStep.phase

  const steps: PlanStep[] = progressSteps.map((s, i) => {
    if (s.status === "completed") return { label: s.label, status: "completed" as const }
    if (s.status === "active") {
      for (let j = 0; j < i; j++) {
        if (steps[j]!.status !== "completed") {
          steps[j] = { label: steps[j]!.label, status: "completed" }
        }
      }
      return { label: s.label, status: "active" as const }
    }
    return { label: s.label, status: "active" as const }
  })

  return { phase, label: PHASE_LABELS[phase] ?? phase, steps }
}

export function derivePlan(messages: ChatMessage[]): PhasePlan | null {
  return deriveDynamicPlan(messages) ?? derivePhasePlan(messages)
}

/** @deprecated Use derivePlan instead */
export function derivePlanSteps(messages: ChatMessage[]): PlanStep[] {
  const plan = derivePlan(messages)
  return plan?.steps ?? []
}
