import { describe, it, expect } from "vitest"
import type { ChatMessage } from "../types"
import { derivePlan } from "./derive-plan-steps"

function sig(phase: string, step: string): ChatMessage {
  return {
    id: Math.random().toString(36),
    role: "assistant",
    content: "",
    timestamp: "",
    toolResults: [
      { name: "signal_phase", result: JSON.stringify({ phase, step }), collapsed: false },
    ],
    proposedAction: null,
    optionCards: [],
  }
}

describe("derivePlan", () => {
  it("labels the eval phase as Evaluation once eval is signalled", () => {
    const plan = derivePlan([sig("eval", "step-1-dataset")])
    expect(plan?.phase).toBe("eval")
    expect(plan?.label).toBe("Evaluation")
  })

  it("does not leave the bar on training after eval starts", () => {
    const plan = derivePlan([
      sig("training", "phase-3-validate"),
      sig("eval", "step-1-dataset"),
    ])
    expect(plan?.label).toBe("Evaluation")
  })

  it("marks every step completed when the final phase is terminal", () => {
    const plan = derivePlan([
      sig("eval", "step-1-dataset"),
      sig("eval", "step-6-complete"),
    ])
    expect(plan?.steps.every((s) => s.status === "completed")).toBe(true)
  })

  it("keeps the last step active mid-workflow", () => {
    const plan = derivePlan([sig("eval", "step-1-dataset")])
    expect(plan?.steps.at(-1)?.status).toBe("active")
  })

  it("marks the phase complete on a job-completion turn even without a terminal signal", () => {
    const completion: ChatMessage = { ...sig("eval", "step-4-validate"), terminal: true }
    const plan = derivePlan([sig("eval", "step-1-dataset"), completion])
    expect(plan?.steps.every((s) => s.status === "completed")).toBe(true)
  })

  it("does not mark complete when a terminal turn also advances to a new phase", () => {
    // SDG finished but the same turn moved on to training — training still in progress.
    const advance: ChatMessage = { ...sig("training", "phase-2-gather"), terminal: true }
    const plan = derivePlan([sig("sdg", "phase-3-validate"), advance])
    expect(plan?.phase).toBe("training")
    expect(plan?.steps.some((s) => s.status === "active")).toBe(true)
    expect(plan?.steps.every((s) => s.status === "completed")).toBe(false)
  })

  it("resets completion when a new run starts in the same phase", () => {
    // SDG run 1 finishes (terminal), then a second SDG run begins — the bar must reflect the
    // new run in progress, not inherit run 1's completed steps.
    const done1: ChatMessage = { ...sig("sdg", "phase-3-validate"), terminal: true }
    const plan = derivePlan([
      sig("sdg", "phase-2-gather"),
      done1,
      sig("sdg", "phase-2-gather"),
    ])
    expect(plan?.phase).toBe("sdg")
    expect(plan?.steps.some((s) => s.status === "active")).toBe(true)
    expect(plan?.steps.every((s) => s.status === "completed")).toBe(false)
  })
})
