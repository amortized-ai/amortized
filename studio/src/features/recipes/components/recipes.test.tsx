import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { RecipeTable } from "./recipe-table"
import { RecipeBuilderForm } from "./recipe-builder-form"
import type { Recipe, DatasetRecord, ModelRecord } from "@/types/api"
import type { RecipeFormState } from "../hooks/use-recipe-state"
import {
  formToJson,
  jsonToForm,
  DEFAULT_FORM,
} from "../hooks/use-recipe-state"

function makeRecipe(overrides: Partial<Recipe> = {}): Recipe {
  return {
    name: "test-recipe",
    type: "training",
    description: "A test recipe",
    version: "1.0",
    schema: {},
    defaults: {},
    ...overrides,
  }
}

function makeDataset(runId: string, name: string): DatasetRecord {
  return {
    run_id: runId,
    name,
    run_name: name,
    experiment_id: "exp-1",
    artifact_uri: `s3://bucket/${runId}`,
    created_at: Date.now(),
    metrics: {},
    params: {},
    tags: {},
  }
}

function makeModel(name: string): ModelRecord {
  return {
    name,
    version: "1",
    run_id: "run-1",
    source: `s3://bucket/models/${name}`,
    created_at: Date.now(),
    description: "",
    aliases: [],
    tags: {},
  }
}

function defaultForm(): RecipeFormState {
  return { ...DEFAULT_FORM }
}

describe("RecipeTable", () => {
  it("renders recipe rows with correct data", () => {
    const recipes = [
      makeRecipe({ name: "recipe-1", type: "training" }),
      makeRecipe({ name: "recipe-2", type: "sdg", description: "SDG recipe" }),
    ]

    render(
      <RecipeTable
        recipes={recipes}
        page={0}
        onPageChange={vi.fn()}
        onSelectRecipe={vi.fn()}
      />,
    )

    expect(screen.getByText("Recipe 1")).toBeInTheDocument()
    expect(screen.getByText("Recipe 2")).toBeInTheDocument()
    expect(screen.getByText("Training")).toBeInTheDocument()
    expect(screen.getByText("SDG")).toBeInTheDocument()
    expect(screen.getByText("SDG recipe")).toBeInTheDocument()
  })

  it("calls onSelectRecipe when row is clicked", () => {
    const onSelect = vi.fn()
    const recipe = makeRecipe({ name: "click-me" })
    render(
      <RecipeTable
        recipes={[recipe]}
        page={0}
        onPageChange={vi.fn()}
        onSelectRecipe={onSelect}
      />,
    )
    fireEvent.click(screen.getByText("Click Me").closest("tr")!)
    expect(onSelect).toHaveBeenCalledWith(recipe)
  })

  it("shows empty state when no recipes", () => {
    render(
      <RecipeTable
        recipes={[]}
        page={0}
        onPageChange={vi.fn()}
        onSelectRecipe={vi.fn()}
      />,
    )
    expect(screen.getByText("No recipes yet")).toBeInTheDocument()
  })
})

describe("RecipeBuilderForm", () => {
  const datasets = [makeDataset("d1", "Dataset 1")]
  const models = [makeModel("Model 1")]

  it("renders all common fields", () => {
    render(
      <MemoryRouter>
        <RecipeBuilderForm
          form={defaultForm()}
          onFieldChange={vi.fn()}
          datasets={datasets}
          models={models}
        />
      </MemoryRouter>,
    )

    expect(screen.getByTestId("recipe-name-input")).toBeInTheDocument()
    expect(screen.getByTestId("recipe-type-select")).toBeInTheDocument()
    expect(screen.getByTestId("recipe-description-input")).toBeInTheDocument()
  })

  it("form container is centered with max-width", () => {
    render(
      <MemoryRouter>
        <RecipeBuilderForm
          form={defaultForm()}
          onFieldChange={vi.fn()}
          datasets={datasets}
          models={models}
        />
      </MemoryRouter>,
    )

    const container = screen.getByTestId("recipe-builder-form")
    expect(container.className).toContain("mx-auto")
    expect(container.className).toContain("max-w-2xl")
  })

  it("renders training sections by default", () => {
    render(
      <MemoryRouter>
        <RecipeBuilderForm
          form={defaultForm()}
          onFieldChange={vi.fn()}
          datasets={datasets}
          models={models}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText("Training Method")).toBeInTheDocument()
    expect(screen.getByText("Model Selection")).toBeInTheDocument()
    expect(screen.getByText("Data")).toBeInTheDocument()
    expect(screen.getByText("Advanced Training Settings")).toBeInTheDocument()
  })

  it("renders SDG sections when type is sdg", () => {
    const form = { ...defaultForm(), type: "sdg" as const }
    render(
      <MemoryRouter>
        <RecipeBuilderForm
          form={form}
          onFieldChange={vi.fn()}
          datasets={datasets}
          models={models}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText("Teacher Model")).toBeInTheDocument()
    expect(screen.getByText("Generation Settings")).toBeInTheDocument()
    expect(screen.getByText("Input Data")).toBeInTheDocument()
  })

  it("calls onFieldChange when training method is selected", () => {
    const onChange = vi.fn()
    render(
      <MemoryRouter>
        <RecipeBuilderForm
          form={defaultForm()}
          onFieldChange={onChange}
          datasets={datasets}
          models={models}
        />
      </MemoryRouter>,
    )

    fireEvent.click(screen.getByTestId("method-dpo"))
    expect(onChange).toHaveBeenCalledWith("training_method", "dpo")
  })

  it("calls onFieldChange when recipe name changes", () => {
    const onChange = vi.fn()
    render(
      <MemoryRouter>
        <RecipeBuilderForm
          form={defaultForm()}
          onFieldChange={onChange}
          datasets={datasets}
          models={models}
        />
      </MemoryRouter>,
    )

    fireEvent.change(screen.getByTestId("recipe-name-input"), {
      target: { value: "new-name" },
    })
    expect(onChange).toHaveBeenCalledWith("name", "new-name")
  })
})

describe("Form <-> JSON sync (reducer logic)", () => {
  it("formToJson produces correct training config", () => {
    const form: RecipeFormState = {
      ...defaultForm(),
      name: "my-recipe",
      type: "training",
      training_method: "sft",
      base_model: "m1",
      training_dataset: "d1",
      validation_dataset: "",
      learning_rate: "2e-5",
      epochs: "3",
      batch_size: "8",
      lora_rank: "16",
      lora_alpha: "32",
    }

    const json = formToJson(form)
    expect(json.name).toBe("my-recipe")
    expect(json.type).toBe("training")
    expect(json.training_method).toBe("sft")
    expect(json.base_model).toBe("m1")
    expect(json.learning_rate).toBe(2e-5)
    expect(json.epochs).toBe(3)
    expect(json.validation_dataset).toBeUndefined()
  })

  it("formToJson produces correct SDG config", () => {
    const form: RecipeFormState = {
      ...defaultForm(),
      type: "sdg",
      teacher_model: "m1",
      num_samples: "50",
      strategy_params: '{"key": "val"}',
      input_data: "d1",
    }

    const json = formToJson(form)
    expect(json.type).toBe("sdg")
    expect(json.teacher_model).toBe("m1")
    expect(json.num_samples).toBe(50)
    expect(json.strategy_params).toEqual({ key: "val" })
  })

  it("jsonToForm round-trips training config", () => {
    const original: RecipeFormState = {
      ...defaultForm(),
      name: "test",
      type: "training",
      training_method: "dpo",
      base_model: "m1",
      learning_rate: "1e-4",
      epochs: "5",
    }

    const json = formToJson(original)
    const restored = jsonToForm(json as Record<string, unknown>, defaultForm())
    expect(restored.name).toBe("test")
    expect(restored.training_method).toBe("dpo")
    expect(restored.learning_rate).toBe("0.0001")
    expect(restored.epochs).toBe("5")
  })

  it("jsonToForm handles invalid JSON fields gracefully", () => {
    const json = { name: "test", type: "training", unknown_field: true }
    const form = jsonToForm(json as Record<string, unknown>, defaultForm())
    expect(form.name).toBe("test")
    expect(form.type).toBe("training")
  })

  it("invalid JSON string produces error", () => {
    let jsonError: string | null = null
    try {
      JSON.parse("{ invalid json }")
    } catch (e) {
      jsonError = (e as Error).message
    }
    expect(jsonError).not.toBeNull()
  })
})
