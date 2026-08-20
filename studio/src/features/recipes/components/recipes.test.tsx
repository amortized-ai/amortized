import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { RecipeTable } from "./recipe-table"
import { RecipeConfigForm } from "./recipe-config-form"
import { buildRecipeEntries } from "../lib/format"
import type { Job } from "@/types/api"
import type { DatasetInfo } from "../lib/format"

function makeJob(overrides: Partial<Job> = {}): Job {
  return {
    id: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    type: "training",
    status: "succeeded",
    config: { algorithm: "lora_sft", model_name_or_path: "Qwen/Qwen3-1.5B" },
    metadata: {},
    recipe: "",
    user_id: null,
    k8s_job_name: null,
    k8s_namespace: null,
    mlflow_run_id: null,
    mlflow_experiment: null,
    parent_job_id: null,
    error: null,
    created_at: "2025-01-01T00:00:00Z",
    started_at: null,
    completed_at: null,
    ...overrides,
  }
}

describe("buildRecipeEntries", () => {
  it("deduplicates jobs with identical configs", () => {
    const jobs = [
      makeJob({ id: "job-1", created_at: "2025-01-01T00:00:00Z" }),
      makeJob({ id: "job-2", created_at: "2025-01-02T00:00:00Z" }),
    ]
    const entries = buildRecipeEntries(jobs, new Map())
    expect(entries).toHaveLength(1)
    expect(entries[0]!.job.id).toBe("job-1")
  })

  it("keeps jobs with different configs", () => {
    const jobs = [
      makeJob({ id: "job-1", config: { algorithm: "sft", model_name_or_path: "m1" } }),
      makeJob({ id: "job-2", config: { algorithm: "dpo", model_name_or_path: "m1" } }),
    ]
    const entries = buildRecipeEntries(jobs, new Map())
    expect(entries).toHaveLength(2)
  })

  it("uses dataset topic for SDG recipe names", () => {
    const jobs = [
      makeJob({ id: "sdg-1", type: "sdg", config: { num_records: 100 } }),
    ]
    const dsMap = new Map<string, DatasetInfo>([
      ["sdg-1", { name: "sdg-abc", topic: "Bank Loans", samples: "100" }],
    ])
    const entries = buildRecipeEntries(jobs, dsMap)
    expect(entries[0]!.name).toBe("sdg-abc - Bank Loans (sdg)")
  })

  it("adds version suffix when names collide", () => {
    const jobs = [
      makeJob({ id: "j1", type: "sdg", config: { num_records: 50 }, created_at: "2025-01-01T00:00:00Z" }),
      makeJob({ id: "j2", type: "sdg", config: { num_records: 100 }, created_at: "2025-01-02T00:00:00Z" }),
    ]
    const dsMap = new Map<string, DatasetInfo>([
      ["j1", { name: "same-ds", topic: "Same Topic", samples: "50" }],
      ["j2", { name: "same-ds", topic: "Same Topic", samples: "100" }],
    ])
    const entries = buildRecipeEntries(jobs, dsMap)
    expect(entries.map((e) => e.name)).toContain("same-ds - Same Topic (sdg) v1")
    expect(entries.map((e) => e.name)).toContain("same-ds - Same Topic (sdg) v2")
  })
})

describe("RecipeTable", () => {
  it("renders recipe entries and calls onSelectJob on click", () => {
    const job = makeJob()
    const entries = buildRecipeEntries([job], new Map())
    const onSelect = vi.fn()

    render(
      <RecipeTable
        recipes={entries}
        page={0}
        onPageChange={vi.fn()}
        onSelectJob={onSelect}
      />,
    )

    expect(screen.getByText("LoRA SFT - Qwen3-1.5B (train)")).toBeInTheDocument()
    fireEvent.click(screen.getByText("LoRA SFT - Qwen3-1.5B (train)").closest("tr")!)
    expect(onSelect).toHaveBeenCalledWith(job, "LoRA SFT - Qwen3-1.5B (train)")
  })

  it("shows empty state when no recipes", () => {
    render(
      <RecipeTable
        recipes={[]}
        page={0}
        onPageChange={vi.fn()}
        onSelectJob={vi.fn()}
      />,
    )
    expect(screen.getByText("No recipes yet")).toBeInTheDocument()
  })
})

describe("RecipeConfigForm", () => {
  it("renders training form with model and algorithm fields", () => {
    const config = {
      algorithm: "lora_sft",
      model_name_or_path: "Qwen/Qwen3-1.5B",
      learning_rate: 0.0002,
      num_train_epochs: 3,
    }
    render(
      <RecipeConfigForm type="training" config={config} onChange={vi.fn()} />,
    )
    expect(screen.getByText("Model")).toBeInTheDocument()
    expect(screen.getByText("Training Parameters")).toBeInTheDocument()
    expect(screen.getByDisplayValue("Qwen/Qwen3-1.5B")).toBeInTheDocument()
  })

  it("renders SDG form with generation settings", () => {
    const config = { num_records: 50, columns: [], model_configs: [] }
    render(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} />,
    )
    expect(screen.getByText("Generation Settings")).toBeInTheDocument()
    expect(screen.getByDisplayValue("50")).toBeInTheDocument()
  })
})
