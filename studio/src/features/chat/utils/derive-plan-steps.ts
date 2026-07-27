import type { ChatMessage, PlanStep, PlanPhase, PhasePlan } from "../types"
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

const EVAL_STEPS: StepDef[] = [
  { label: "Choose method", matchSteps: ["understand_task", "load_skill"] },
  { label: "Configure", matchSteps: ["gather_requirements"] },
  { label: "Estimate cost", matchSteps: ["estimate_cost", "confirm"] },
  { label: "Evaluate model", matchSteps: ["execute", "review"] },
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
