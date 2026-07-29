import type { ChatMessage, PlanStep, PlanPhase, PhasePlan, ProgressStep } from "../types"
import { getLatestPhase, parsePhaseTag } from "./workflow-options"

const PHASE_LABELS: Record<PlanPhase, string> = {
  sdg: "Data Generation",
  training: "Model Training",
  eval: "Evaluation",
}

const VALID_PHASES = new Set<string>(["sdg", "training", "eval"])

function isValidPhase(phase: string): phase is PlanPhase {
  return VALID_PHASES.has(phase)
}

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

const EVAL_STEPS: StepDef[] = [
  { label: "Configure evaluation", matchSteps: ["understand_task", "load_skill"] },
  { label: "Run evaluation", matchSteps: ["execute", "review"] },
]

const PHASE_CONFIG: Record<PlanPhase, { label: string; steps: StepDef[] }> = {
  sdg: { label: "Data Generation", steps: SDG_STEPS },
  training: { label: "Model Training", steps: TRAINING_STEPS },
  eval: { label: "Evaluation", steps: EVAL_STEPS },
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

/** @deprecated Use derivePhasePlan instead */
export function derivePlanSteps(messages: ChatMessage[]): PlanStep[] {
  const plan = derivePhasePlan(messages)
  return plan?.steps ?? []
}

export function deriveDynamicPlan(messages: ChatMessage[]): PhasePlan | null {
  const steps = new Map<string, ProgressStep>()
  let latestPhase: PlanPhase | null = null

  for (const msg of messages) {
    for (const tool of msg.toolResults) {
      if (tool.name !== "signal_progress" && tool.name !== "signal progress") continue
      try {
        const data = typeof tool.result === "string" ? JSON.parse(tool.result) : tool.result
        if (!data?.step_id || !data?.label) continue
        const phase = data.phase as string
        if (!phase || !isValidPhase(phase)) continue
        latestPhase = phase

        if (data.status === "active") {
          for (const [, existing] of steps) {
            if (existing.status === "active") existing.status = "completed"
          }
        }

        steps.set(data.step_id, {
          phase,
          stepId: data.step_id,
          label: data.label,
          status: data.status === "completed" ? "completed" : "active",
        })
      } catch { /* ignore parse errors */ }
    }
  }

  if (steps.size === 0 || !latestPhase) return null

  const planSteps: PlanStep[] = [...steps.values()].map((s) => ({
    label: s.label,
    status: s.status,
  }))

  return {
    phase: latestPhase,
    label: PHASE_LABELS[latestPhase],
    steps: planSteps,
  }
}

export function derivePlan(messages: ChatMessage[]): PhasePlan | null {
  return deriveDynamicPlan(messages) ?? derivePhasePlan(messages)
}
