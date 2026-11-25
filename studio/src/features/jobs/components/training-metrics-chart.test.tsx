import { render, screen } from "@testing-library/react"
import { describe, it, expect } from "vitest"
import { TrainingMetricsChart } from "./training-metrics-chart"
import { isGrpoAlgorithm } from "../lib/metric-aliases"
import type { AllMetricsData } from "../api/use-jobs"

const history = (values: number[]) => values.map((value, i) => ({ step: i + 1, value }))

// Shape of a verl (lora_grpo) run: namespaced keys, no plain `loss`.
const grpoData: AllMetricsData = {
  histories: {
    "critic/rewards/mean": history([0.1, 0.2, 0.3125]),
    "actor/entropy": history([0.3, 0.2, 0.1676]),
    "actor/grad_norm": history([0.05, 0.03, 0.023]),
    "actor/lr": history([1e-6, 1e-6, 1e-6]),
  },
  finalValues: {
    "critic/rewards/mean": 0.3125,
    "actor/entropy": 0.1676,
    "actor/lr": 1e-6,
    "training/global_step": 13,
    "actor/loss": 0,
    train_runtime: 125,
  },
  params: {},
}

const sftData: AllMetricsData = {
  histories: { loss: history([2.5, 2.3]) },
  finalValues: { loss: 2.3, learning_rate: 2e-5, epoch: 1, train_runtime: 60 },
  params: {},
}

describe("TrainingMetricsChart", () => {
  it("shows reward-centric cards and charts for lora_grpo", () => {
    render(<TrainingMetricsChart data={grpoData} isLoading={false} algorithm="lora_grpo" />)

    expect(screen.getByText("Final Reward")).toBeInTheDocument()
    expect(screen.getByText("0.3125")).toBeInTheDocument()
    expect(screen.getByText("1.00e-6")).toBeInTheDocument()
    expect(screen.getByText("2m 5s")).toBeInTheDocument()
    expect(screen.getByText("13")).toBeInTheDocument()
    expect(screen.getByText("Mean Reward")).toBeInTheDocument()
    expect(screen.getByText("Policy Entropy")).toBeInTheDocument()
    expect(screen.queryByText("Final Loss")).not.toBeInTheDocument()
  })

  it("keeps the loss layout for non-GRPO algorithms", () => {
    render(<TrainingMetricsChart data={sftData} isLoading={false} algorithm="lora_sft" />)

    expect(screen.getByText("Final Loss")).toBeInTheDocument()
    expect(screen.getByText("2.3000")).toBeInTheDocument()
    expect(screen.getByText("Training Loss")).toBeInTheDocument()
    expect(screen.queryByText("Final Reward")).not.toBeInTheDocument()
  })

  it("keeps the loss layout when the algorithm is unknown", () => {
    render(<TrainingMetricsChart data={sftData} isLoading={false} />)

    expect(screen.getByText("Final Loss")).toBeInTheDocument()
  })
})

describe("isGrpoAlgorithm", () => {
  it("accepts the GRPO spellings and rejects others", () => {
    expect(isGrpoAlgorithm("lora_grpo")).toBe(true)
    expect(isGrpoAlgorithm(" lora-grpo ")).toBe(true)
    expect(isGrpoAlgorithm("grpo")).toBe(true)
    expect(isGrpoAlgorithm("lora_sft")).toBe(false)
    expect(isGrpoAlgorithm(undefined)).toBe(false)
  })
})
