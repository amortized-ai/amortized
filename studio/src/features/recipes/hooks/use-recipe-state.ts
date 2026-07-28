import { useReducer, useCallback } from "react"
import type { Recipe } from "@/types/api"

export type RecipeType = "training" | "sdg"
export type TrainingMethod = "lora_sft" | "sft" | "osft" | "dpo" | "grpo" | "lora_grpo" | "kto" | "gepa" | "gkd"

export interface RecipeFormState {
  name: string
  type: RecipeType
  description: string

  // Training
  training_method: TrainingMethod
  base_model: string
  training_dataset: string
  validation_dataset: string
  learning_rate: string
  epochs: string
  batch_size: string
  lora_rank: string
  lora_alpha: string

  // SDG
  teacher_model: string
  num_samples: string
  strategy_params: string
  input_data: string

}

export interface RecipeState {
  form: RecipeFormState
  json: string
  jsonError: string | null
  isDirty: boolean
  originalName: string | null
}

type RecipeAction =
  | { type: "SET_FIELD"; field: keyof RecipeFormState; value: string }
  | { type: "SET_JSON"; json: string }
  | { type: "LOAD_RECIPE"; recipe: Recipe }
  | { type: "RESET"; recipeType?: RecipeType }
  | { type: "SET_NAME"; name: string }

const DEFAULT_FORM: RecipeFormState = {
  name: "",
  type: "training",
  description: "",
  training_method: "sft",
  base_model: "",
  training_dataset: "",
  validation_dataset: "",
  learning_rate: "2e-5",
  epochs: "3",
  batch_size: "8",
  lora_rank: "16",
  lora_alpha: "32",
  teacher_model: "",
  num_samples: "100",
  strategy_params: "{}",
  input_data: "",
}

function formToJson(form: RecipeFormState): Record<string, unknown> {
  const base: Record<string, unknown> = {
    name: form.name,
    type: form.type,
    description: form.description,
  }

  switch (form.type) {
    case "training":
      return {
        ...base,
        training_method: form.training_method,
        base_model: form.base_model,
        training_dataset: form.training_dataset,
        validation_dataset: form.validation_dataset || undefined,
        learning_rate: parseFloat(form.learning_rate) || 2e-5,
        epochs: parseInt(form.epochs, 10) || 3,
        batch_size: parseInt(form.batch_size, 10) || 8,
        lora_rank: parseInt(form.lora_rank, 10) || 16,
        lora_alpha: parseInt(form.lora_alpha, 10) || 32,
      }
    case "sdg":
      return {
        ...base,
        teacher_model: form.teacher_model,
        num_samples: parseInt(form.num_samples, 10) || 100,
        strategy_params: safeParseJson(form.strategy_params, {}),
        input_data: form.input_data,
      }
  }
}

function safeParseJson(str: string, fallback: unknown): unknown {
  try {
    return JSON.parse(str)
  } catch {
    return fallback
  }
}

function jsonToForm(
  obj: Record<string, unknown>,
  currentForm: RecipeFormState,
): RecipeFormState {
  const type = (obj.type as RecipeType) ?? currentForm.type

  return {
    name: (obj.name as string) ?? currentForm.name,
    type,
    description: (obj.description as string) ?? currentForm.description,
    training_method:
      (obj.training_method as TrainingMethod) ?? currentForm.training_method,
    base_model: (obj.base_model as string) ?? currentForm.base_model,
    training_dataset:
      (obj.training_dataset as string) ?? currentForm.training_dataset,
    validation_dataset:
      (obj.validation_dataset as string) ?? currentForm.validation_dataset,
    learning_rate: obj.learning_rate != null
      ? String(obj.learning_rate)
      : currentForm.learning_rate,
    epochs: obj.epochs != null ? String(obj.epochs) : currentForm.epochs,
    batch_size: obj.batch_size != null
      ? String(obj.batch_size)
      : currentForm.batch_size,
    lora_rank: obj.lora_rank != null
      ? String(obj.lora_rank)
      : currentForm.lora_rank,
    lora_alpha: obj.lora_alpha != null
      ? String(obj.lora_alpha)
      : currentForm.lora_alpha,
    teacher_model:
      (obj.teacher_model as string) ?? currentForm.teacher_model,
    num_samples: obj.num_samples != null
      ? String(obj.num_samples)
      : currentForm.num_samples,
    strategy_params:
      obj.strategy_params != null
        ? JSON.stringify(obj.strategy_params, null, 2)
        : currentForm.strategy_params,
    input_data: (obj.input_data as string) ?? currentForm.input_data,
  }
}

function reducer(state: RecipeState, action: RecipeAction): RecipeState {
  switch (action.type) {
    case "SET_FIELD": {
      const form = { ...state.form, [action.field]: action.value }
      const jsonObj = formToJson(form)
      return {
        ...state,
        form,
        json: JSON.stringify(jsonObj, null, 2),
        jsonError: null,
        isDirty: true,
      }
    }
    case "SET_JSON": {
      try {
        const obj = JSON.parse(action.json) as Record<string, unknown>
        const form = jsonToForm(obj, state.form)
        return {
          ...state,
          form,
          json: action.json,
          jsonError: null,
          isDirty: true,
        }
      } catch (e) {
        return {
          ...state,
          json: action.json,
          jsonError: (e as Error).message,
          isDirty: true,
        }
      }
    }
    case "LOAD_RECIPE": {
      const recipe = action.recipe
      const merged = { ...recipe.defaults, name: recipe.name, type: recipe.type, description: recipe.description }
      const form = jsonToForm(merged as Record<string, unknown>, DEFAULT_FORM)
      const jsonObj = formToJson(form)
      return {
        form,
        json: JSON.stringify(jsonObj, null, 2),
        jsonError: null,
        isDirty: false,
        originalName: recipe.name,
      }
    }
    case "RESET": {
      const form = { ...DEFAULT_FORM, type: action.recipeType ?? "training" }
      const jsonObj = formToJson(form)
      return {
        form,
        json: JSON.stringify(jsonObj, null, 2),
        jsonError: null,
        isDirty: false,
        originalName: null,
      }
    }
    case "SET_NAME": {
      const form = { ...state.form, name: action.name }
      const jsonObj = formToJson(form)
      return {
        ...state,
        form,
        json: JSON.stringify(jsonObj, null, 2),
        jsonError: null,
        isDirty: true,
      }
    }
  }
}

function getInitialState(): RecipeState {
  const form = { ...DEFAULT_FORM }
  const jsonObj = formToJson(form)
  return {
    form,
    json: JSON.stringify(jsonObj, null, 2),
    jsonError: null,
    isDirty: false,
    originalName: null,
  }
}

export function useRecipeState() {
  const [state, dispatch] = useReducer(reducer, undefined, getInitialState)

  const setField = useCallback(
    (field: keyof RecipeFormState, value: string) =>
      dispatch({ type: "SET_FIELD", field, value }),
    [],
  )

  const setJson = useCallback(
    (json: string) => dispatch({ type: "SET_JSON", json }),
    [],
  )

  const loadRecipe = useCallback(
    (recipe: Recipe) => dispatch({ type: "LOAD_RECIPE", recipe }),
    [],
  )

  const reset = useCallback(
    (recipeType?: RecipeType) => dispatch({ type: "RESET", recipeType }),
    [],
  )

  const getConfig = useCallback(
    () => formToJson(state.form),
    [state.form],
  )

  return { state, setField, setJson, loadRecipe, reset, getConfig }
}

export { formToJson, jsonToForm, DEFAULT_FORM }
