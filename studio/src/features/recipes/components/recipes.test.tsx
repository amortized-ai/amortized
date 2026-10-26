import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { RecipeTable } from "./recipe-table"
import { RecipeConfigForm } from "./recipe-config-form"
import { buildRecipeEntries } from "../lib/format"
import type { Job } from "@/types/api"
import type { DatasetInfo } from "../lib/format"

vi.mock("@/features/documents/api/use-documents", () => ({
  useDocuments: () => ({
    data: [
      { document_id: "d1", filename: "Doc One", format: "pdf", mlflow_run_id: null, created_at: null },
      { document_id: "d2", filename: "Doc Two", format: "pdf", mlflow_run_id: null, created_at: null },
    ],
  }),
}))

vi.mock("@/features/datasets/api/use-datasets", () => ({
  useDatasets: () => ({
    data: [{ run_id: "r1", name: "My Dataset", tags: { job_id: "job-xyz" } }],
  }),
}))

vi.mock("@/features/recipes/api/use-gateway-models", () => ({
  useGatewayModels: () => ({
    data: {
      models: [
        { name: "gpt-oss", provider: "gateway", model_name: "openai/gpt-oss-120b" },
        { name: "gpt-4o", provider: "openai", model_name: "gpt-4o" },
      ],
      gateway_url: "",
    },
  }),
}))

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

  it("marks required fields and opens their section", () => {
    const schema = {
      sdg: { required: ["columns"], properties: { columns: { description: "Pipeline columns" } } },
    }
    const config = { columns: [{ column_type: "sampler", name: "topic" }] }
    render(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} schema={schema} />,
    )
    // required marker rendered on the Columns section header
    expect(screen.getAllByLabelText("required").length).toBeGreaterThan(0)
    // section is open by default → its item is visible
    expect(screen.getByText("topic")).toBeInTheDocument()
  })

  it("marks a per-recipe requiredKey (document_ids) as required", () => {
    const config = { document_ids: [] }
    render(
      <RecipeConfigForm
        type="sdg"
        config={config}
        onChange={vi.fn()}
        requiredKeys={["document_ids"]}
      />,
    )
    // Document Ids section header shows the required marker (from requiredKeys,
    // not the schema).
    expect(screen.getAllByLabelText("required").length).toBeGreaterThan(0)
    expect(screen.getAllByText("Document Ids").length).toBeGreaterThan(0)
  })

  it("opens a required section when the schema loads after mount (cold load)", () => {
    const config = { columns: [{ column_type: "sampler", name: "topic" }] }
    // Mount before the schema is available (async useQuery not yet resolved).
    const { rerender } = render(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} />,
    )
    expect(screen.queryByText("topic")).not.toBeInTheDocument()
    // Schema arrives → the required Columns section must open, not stay collapsed.
    const schema = { sdg: { required: ["columns"], properties: { columns: {} } } }
    rerender(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} schema={schema} />,
    )
    expect(screen.getByText("topic")).toBeInTheDocument()
  })

  it("renders a document picker for document_ids and toggling updates the array", () => {
    const onChange = vi.fn()
    const config = { document_ids: ["d1"] }
    render(
      <RecipeConfigForm type="sdg" config={config} onChange={onChange} />,
    )
    // selected document shown by filename (not raw id)
    expect(screen.getAllByText("Doc One").length).toBeGreaterThan(0)
    // list is collapsed by default
    expect(screen.queryByText("Doc Two")).not.toBeInTheDocument()
    // expand the collapsible → other documents are offered
    fireEvent.click(screen.getByText("1 selected"))
    expect(screen.getByText("Doc Two")).toBeInTheDocument()
    // removing the selection bubbles the updated config up
    fireEvent.click(screen.getByLabelText("Remove Doc One"))
    expect(onChange).toHaveBeenCalledWith({ document_ids: [] })
  })

  it("renders a dataset picker for parent_job_id, labelled 'Dataset'", () => {
    const config = { parent_job_id: "" }
    render(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} />,
    )
    // Labelled "Dataset", not the raw "Parent Job Id" field name.
    expect(screen.getByText("Dataset")).toBeInTheDocument()
    expect(screen.queryByText("Parent Job Id")).not.toBeInTheDocument()
    expect(screen.getByText("Select a dataset...")).toBeInTheDocument()
  })

  it("flags the training dataset (parent_job_id) required via requiredKeys", () => {
    // The dataset lives in the existing parent_job_id picker ("Other Parameters");
    // requiredKeys from the template _meta flags it required and opens the section.
    const config = { algorithm: "osft", model_name_or_path: "Qwen/Qwen3.5-4B", parent_job_id: "" }
    render(
      <RecipeConfigForm
        type="training"
        config={config}
        onChange={vi.fn()}
        requiredKeys={["parent_job_id"]}
      />,
    )
    // The dataset picker is shown (labelled "Dataset") and marked required.
    expect(screen.getByText("Dataset")).toBeInTheDocument()
    expect(screen.getByText("Select a dataset...")).toBeInTheDocument()
    expect(screen.getAllByLabelText("required").length).toBeGreaterThan(0)
  })

  it("renders a catalog model picker for model_configs entries", () => {
    const schema = { sdg: { required: ["model_configs"] } }
    const config = { model_configs: [{ alias: "text", model: "gpt-4o", provider: "openai" }] }
    render(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} schema={schema} />,
    )
    // model_configs section opened (required) → the model picker renders
    expect(screen.getByText("Model")).toBeInTheDocument()
    // the derived provider is shown read-only (no separate editable field)
    expect(screen.getByText("Provider:")).toBeInTheDocument()
    expect(screen.getByText("openai")).toBeInTheDocument()
  })

  it("renders a model_alias dropdown for columns", () => {
    const schema = { sdg: { required: ["columns"] } }
    const config = {
      model_configs: [{ alias: "text", model: "gpt-4o", provider: "openai" }],
      columns: [{ column_type: "llm-text", name: "answer", model_alias: "text" }],
    }
    render(
      <RecipeConfigForm type="sdg" config={config} onChange={vi.fn()} schema={schema} />,
    )
    expect(screen.getByText("Model Alias")).toBeInTheDocument()
  })
})
