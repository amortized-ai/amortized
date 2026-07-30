import type { ChatMessage, PlanStep, PlanPhase, PhasePlan } from "../types"
import { getLatestPhase, parsePhaseTag } from "./workflow-options"

const PHASE_LABELS: Record<PlanPhase, string> = {
  sdg: "Data Generation",
  training: "Model Training",
  eval: "Evaluation",
}

interface ProgressEntry {
  stepId: string
  label: string
  status: "active" | "completed"
  phase: PlanPhase
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
  const steps = new Map<string, ProgressEntry>()
  let latestPhase: PlanPhase | null = null

  for (const msg of messages) {
    for (const tool of msg.toolResults) {
      if (tool.name !== "signal_progress" && tool.name !== "signal progress") continue
      const data = tryParseToolResult(tool.result)
      if (!data) continue
      const phase = String(data.phase ?? "")
      if (!isValidPhase(phase)) continue
      const stepId = String(data.step_id ?? "")
      const label = String(data.label ?? "")
      const status = data.status === "completed" ? "completed" : "active"
      if (!stepId || !label) continue

      if (latestPhase && phase !== latestPhase) {
        steps.clear()
      }
      latestPhase = phase
      steps.set(stepId, { stepId, label, status, phase })
    }
  }

  if (!latestPhase || steps.size === 0) return null

  const stepList = [...steps.values()]
  const lastActiveIdx = stepList.findLastIndex((s) => s.status === "active")
  if (lastActiveIdx > 0) {
    for (let i = 0; i < lastActiveIdx; i++) {
      stepList[i]!.status = "completed"
    }
  }

  return {
    phase: latestPhase,
    label: PHASE_LABELS[latestPhase],
    steps: stepList.map((s): PlanStep => ({ label: s.label, status: s.status })),
  }
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
