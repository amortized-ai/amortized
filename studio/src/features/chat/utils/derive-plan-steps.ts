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

// signal_phase now reports the workflow's own section slug (one file per section),
// not the old coarse 7-step enum. Map each slug to the bar label its section rolls up
// to; sibling eval sections (dataset / held-out, report / complete) share a label so
// they collapse to one bar stage.
const PHASE_STEP_LABELS: Record<string, string> = {
  // sdg / training
  "phase-1-route": "Routing to the right guide",
  "phase-2-gather": "Gathering requirements",
  "phase-3-validate": "Reviewing configuration",
  "phase-4-complete": "Checking results",
  // eval
  "step-0-what-to-eval": "Confirming the model",
  "step-1-dataset": "Choosing the eval dataset",
  "held-out-ask-count": "Choosing the eval dataset",
  "held-out-request-help": "Choosing the eval dataset",
  "step-1_5-how-much": "Confirming the sample size",
  "step-2-model-source": "Choosing the model source",
  "step-3-metrics": "Designing metrics",
  "step-4-validate": "Reviewing configuration",
  "step-5-report": "Checking results",
  "step-6-complete": "Checking results",
}

// The phase slugs that mean the workflow has reached its end — once signalled, every
// bar step is marked complete so the bar doesn't spin. "review" is kept for back-compat
// with any transcript signalled under the old enum.
const TERMINAL_STEPS = new Set(["review", "phase-4-complete", "step-6-complete"])

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
    const done = TERMINAL_STEPS.has(latestStep) || terminalSeen
    return {
      phase: latestPhase,
      label: cfg.label,
      steps: cfg.steps.map((def, i): PlanStep => ({
        label: def.label,
        status: done || i < activeIdx ? "completed" : i === activeIdx ? "active" : "pending",
      })),
    }
  }

  // When the last signalled step is a terminal phase — or a job-completion turn made the
  // phase terminal — the workflow has finished; mark every step complete so the bar
  // isn't left spinning.
  const done = TERMINAL_STEPS.has(latestStep) || terminalSeen

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
  { label: "Routing to the right guide", matchSteps: ["phase-1-route"] },
  { label: "Gathering requirements", matchSteps: ["phase-2-gather"] },
  { label: "Reviewing configuration", matchSteps: ["phase-3-validate"] },
  { label: "Checking results", matchSteps: ["phase-4-complete"] },
]

const TRAINING_STEPS: StepDef[] = [
  { label: "Routing to the right guide", matchSteps: ["phase-1-route"] },
  { label: "Gathering requirements", matchSteps: ["phase-2-gather"] },
  { label: "Reviewing configuration", matchSteps: ["phase-3-validate"] },
  { label: "Checking results", matchSteps: ["phase-4-complete"] },
]

const EVAL_STEPS: StepDef[] = [
  { label: "Confirming the model", matchSteps: ["step-0-what-to-eval"] },
  {
    label: "Choosing the eval dataset",
    matchSteps: ["step-1-dataset", "held-out-ask-count", "held-out-request-help"],
  },
  { label: "Confirming the sample size", matchSteps: ["step-1_5-how-much"] },
  { label: "Choosing the model source", matchSteps: ["step-2-model-source"] },
  { label: "Designing metrics", matchSteps: ["step-3-metrics"] },
  { label: "Reviewing configuration", matchSteps: ["step-4-validate"] },
  { label: "Checking results", matchSteps: ["step-5-report", "step-6-complete"] },
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
  const isCompleted = TERMINAL_STEPS.has(parsed.step)

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
