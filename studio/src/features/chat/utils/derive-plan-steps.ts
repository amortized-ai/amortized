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
  "create_sdg_job": "Submitting SDG job",
  "create_training_job": "Submitting training job",
  "create_eval_job": "Submitting eval job",
  "get_job_detail": "Checking job status",
  "get_job_logs": "Reading job logs",
}

// Step slugs match the `<!--signal:slug-->` markers in each agent's workflow.md `## Workflow`
// headings (sdg/training are 4-phase; eval is a finer 8-step sequence). "complete" is the
// terminal step for every agent.
const PHASE_STEP_LABELS: Record<string, string> = {
  // sdg / training phases
  "route": "Routing to sub-skill",
  "gather": "Gathering requirements",
  "validate": "Reviewing configuration",
  "complete": "Completing",
  // eval fine steps
  "understand-task": "Understanding your task",
  "eval-dataset": "Confirming the eval dataset",
  "how-much": "Choosing how much to evaluate",
  "model-source": "Choosing the model",
  "metrics": "Designing the metrics",
  "report": "Reporting results",
}

const TERMINAL_STEP = "complete"

export function deriveDynamicPlan(messages: ChatMessage[]): PhasePlan | null {
  let latestPhase: PlanPhase | null = null
  let latestStep = ""
  let terminalSeen = false
  const seenLabels = new Set<string>()
  const steps: Array<{ label: string }> = []

  for (const msg of messages) {
    let phaseChangedThisMsg = false
    for (const tool of msg.toolResults) {
      if (tool.name === "signal_phase" || tool.name === "signal phase") {
        const data = tryParseToolResult(tool.result)
        if (!data) continue
        const phase = String(data.phase ?? "")
        if (!isValidPhase(phase)) continue
        const step = String(data.step ?? "")

        if (latestPhase && (phase !== latestPhase || terminalSeen)) {
          // Reset on a phase change OR a new run in the SAME phase after a terminal one —
          // otherwise a second same-phase run (e.g. a re-run SDG job) inherits the finished
          // run's completed steps.
          seenLabels.clear()
          steps.length = 0
          terminalSeen = false
          if (phase !== latestPhase) phaseChangedThisMsg = true
        }
        latestPhase = phase

        latestStep = step

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
    // A job-completion continuation turn is terminal for the phase it finished —
    // unless the same turn advanced to a new phase (then the new phase leads and is
    // still in progress). Lets the bar settle when the driver skips the "review" signal.
    if (msg.terminal && !phaseChangedThisMsg) terminalSeen = true
  }

  if (!latestPhase) return null

  // The phase is known but no recognizable step was collected — e.g. the server
  // backfilled a phase-only signal because the driver forgot to call signal_phase.
  // Render the static plan for the phase so the bar still shows the right section
  // (this also covers "eval", which parsePhaseTag's static path does not accept).
  if (steps.length === 0) {
    const cfg = STATIC_PHASE_CONFIG[latestPhase]
    if (!cfg) return null
    const activeIdx = resolveStepIndex(cfg.steps, latestStep)
    const done = latestStep === TERMINAL_STEP || terminalSeen
    return {
      phase: latestPhase,
      label: cfg.label,
      steps: cfg.steps.map((def, i): PlanStep => ({
        label: def.label,
        status: done || i < activeIdx ? "completed" : i === activeIdx ? "active" : "pending",
      })),
    }
  }

  // When the last signalled step is the terminal step — or a job-completion turn made the
  // phase terminal — the workflow has finished; mark every step complete so the bar
  // isn't left spinning.
  const done = latestStep === TERMINAL_STEP || terminalSeen

  return {
    phase: latestPhase,
    label: PHASE_LABELS[latestPhase],
    steps: steps.map((s, i): PlanStep => ({
      label: s.label,
      status: done || i < steps.length - 1 ? "completed" : "active",
    })),
  }
}

interface StepDef {
  label: string
  matchSteps: string[]
}

const SDG_STEPS: StepDef[] = [
  { label: "Routing to sub-skill", matchSteps: ["route"] },
  { label: "Gathering requirements", matchSteps: ["gather"] },
  { label: "Reviewing configuration", matchSteps: ["validate"] },
  { label: "Completing", matchSteps: ["complete"] },
]

const TRAINING_STEPS: StepDef[] = [
  { label: "Routing to sub-skill", matchSteps: ["route"] },
  { label: "Gathering requirements", matchSteps: ["gather"] },
  { label: "Reviewing configuration", matchSteps: ["validate"] },
  { label: "Completing", matchSteps: ["complete"] },
]

const EVAL_STEPS: StepDef[] = [
  { label: "Understanding your task", matchSteps: ["understand-task"] },
  { label: "Confirming the eval dataset", matchSteps: ["eval-dataset"] },
  { label: "Choosing how much to evaluate", matchSteps: ["how-much"] },
  { label: "Choosing the model", matchSteps: ["model-source"] },
  { label: "Designing the metrics", matchSteps: ["metrics"] },
  { label: "Reviewing configuration", matchSteps: ["validate"] },
  { label: "Reporting results", matchSteps: ["report"] },
  { label: "Completing", matchSteps: ["complete"] },
]

const STATIC_PHASE_CONFIG: Partial<Record<PlanPhase, { label: string; steps: StepDef[] }>> = {
  sdg: { label: "Data Generation", steps: SDG_STEPS },
  training: { label: "Model Training", steps: TRAINING_STEPS },
  eval: { label: "Evaluation", steps: EVAL_STEPS },
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
  const isCompleted = parsed.step === TERMINAL_STEP

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
