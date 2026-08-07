import { http, HttpResponse } from "msw"
import type {
  ConfigResponse,
  HealthResponse,
  Job,
  MlflowRunsSearchResponse,
  MlflowRegisteredModelsResponse,
  MlflowModelVersionsResponse,
  Recipe,
} from "@/types/api"

const mockJobs: Job[] = [
  {
    id: "job-001",
    type: "training",
    status: "running",
    config: { model: "llama-3.1-8b", method: "lora" },
    metadata: { name: "Fine-tune LLaMA" },
    recipe: "lora-sft",
    user_id: "shiv",
    k8s_job_name: "amortized-job-001",
    k8s_namespace: "amortized-jobs",
    mlflow_run_id: null,
    mlflow_experiment: "amortized/training/job-001",
    parent_job_id: "job-000",
    error: null,
    created_at: "2026-06-01T10:00:00Z",
    started_at: "2026-06-01T10:05:00Z",
    completed_at: null,
  },
  {
    id: "job-002",
    type: "sdg",
    status: "succeeded",
    config: { teacher_model: "gpt-4o-mini", num_samples: 100 },
    metadata: { name: "SDG run" },
    recipe: "",
    user_id: "shiv",
    k8s_job_name: "amortized-job-002",
    k8s_namespace: "amortized-jobs",
    mlflow_run_id: "run-abc-123",
    mlflow_experiment: "amortized/sdg/job-002",
    parent_job_id: null,
    error: null,
    created_at: "2026-05-28T08:00:00Z",
    started_at: "2026-05-28T08:01:00Z",
    completed_at: "2026-05-28T09:30:00Z",
  },
]

const mockRecipes: Recipe[] = [
  {
    name: "lora-sft",
    type: "training",
    description: "LoRA supervised fine-tuning",
    version: "1.0.0",
    schema: {},
    defaults: { lora_rank: 16, learning_rate: 2e-4 },
  },
]

const mockHealth: HealthResponse = {
  status: "ok",
  timestamp: new Date().toISOString(),
}

const mockConfig: ConfigResponse = {
  mlflow_tracking_uri: "http://mlflow:5000",
  mlflow_gateway_uri: "http://mlflow:5000/gateway",
  default_compute_backend: "kubernetes",
  compute_namespace: "amortized-jobs",
  image_registry: "ghcr.io/amortized-ai",
  available_backends: ["kubernetes"],
  version: "1.0.0",
}

const mockMlflowRunsSearch: MlflowRunsSearchResponse = {
  runs: [
    {
      info: {
        run_id: "run-sdg-001",
        experiment_id: "exp-001",
        status: "FINISHED",
        start_time: 1717200000000,
        end_time: 1717200060000,
        artifact_uri: "s3://mlflow/exp-001/run-sdg-001/artifacts",
        run_name: "sdg-ticket-classifier",
      },
      data: {
        metrics: [{ key: "num_samples_generated", value: 200, step: 0, timestamp: 1717200060000 }],
        params: [{ key: "model", value: "gpt-4o-mini" }, { key: "num_samples", value: "200" }],
        tags: [{ key: "job_type", value: "sdg" }, { key: "job_id", value: "job-003" }],
      },
    },
  ],
}

const mockRegisteredModels: MlflowRegisteredModelsResponse = {
  registered_models: [
    {
      name: "ticket-classifier",
      creation_timestamp: 1717200000000,
      last_updated_timestamp: 1717200060000,
      description: "Ticket classification model",
      latest_versions: [
        {
          name: "ticket-classifier",
          version: "1",
          creation_timestamp: 1717200000000,
          last_updated_timestamp: 1717200060000,
          current_stage: "Production",
          source: "runs:/run-train-001/artifacts/model",
          run_id: "run-train-001",
          status: "READY",
          aliases: ["champion"],
        },
      ],
    },
  ],
}

const mockModelVersions: MlflowModelVersionsResponse = {
  model_versions: [
    {
      name: "ticket-classifier",
      version: "1",
      creation_timestamp: 1717200000000,
      last_updated_timestamp: 1717200060000,
      current_stage: "Production",
      source: "runs:/run-train-001/artifacts/model",
      run_id: "run-train-001",
      status: "READY",
      aliases: ["champion"],
    },
  ],
}

const mockDatasets = [
  {
    run_id: "run-sdg-001",
    name: "ticket-classifier-data",
    topic: "Support tickets",
    source: "sdg",
    samples: "200",
    teacher_model: "gpt-4o-mini",
    job_id: "job-002",
    experiment_id: "exp-001",
    created_at: 1717200000000,
  },
]

export const handlers = [
  // Health
  http.get("*/api/v1/health", () => {
    return HttpResponse.json(mockHealth)
  }),

  // Datasets
  http.get("*/api/v1/datasets", () => {
    return HttpResponse.json(mockDatasets)
  }),

  // Config
  http.get("*/api/v1/config", () => {
    return HttpResponse.json(mockConfig)
  }),

  // Jobs
  http.get("*/api/v1/jobs", () => {
    return HttpResponse.json(mockJobs)
  }),

  http.get("*/api/v1/jobs/:id", ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id)
    if (!job) return HttpResponse.json({ detail: "Not found" }, { status: 404 })
    return HttpResponse.json(job)
  }),

  http.get("*/api/v1/jobs/:id/logs", ({ params }) => {
    return HttpResponse.json({
      job_id: params.id,
      logs: ["[2026-06-01 10:05:00] Training started", "[2026-06-01 10:05:01] Epoch 1/3"],
      message: "ok",
    })
  }),

  http.post("*/api/v1/jobs", () => {
    return HttpResponse.json(mockJobs[0], { status: 201 })
  }),

  http.post("*/api/v1/jobs/recipe", () => {
    return HttpResponse.json(mockJobs[0], { status: 201 })
  }),

  http.delete("*/api/v1/jobs/:id", ({ params }) => {
    const job = mockJobs.find((j) => j.id === params.id)
    if (!job) return HttpResponse.json({ detail: "Not found" }, { status: 404 })
    return HttpResponse.json({ ...job, status: "cancelled" })
  }),

  // Recipes
  http.get("*/api/v1/recipes", () => {
    return HttpResponse.json(mockRecipes)
  }),

  http.get("*/api/v1/recipes/:name", ({ params }) => {
    const recipe = mockRecipes.find((r) => r.name === params.name)
    if (!recipe)
      return HttpResponse.json({ detail: "Not found" }, { status: 404 })
    return HttpResponse.json(recipe)
  }),

  // OpenCode agent: providers
  http.get("*/agent/provider", () => {
    return HttpResponse.json({
      all: [
        { id: "google-vertex-anthropic", models: [{ id: "claude-opus-4-6@default" }, { id: "claude-sonnet-4-6@default" }] },
        { id: "anthropic", models: [{ id: "claude-opus-4-6-latest" }, { id: "claude-sonnet-4-6-latest" }] },
        { id: "openai", models: [{ id: "gpt-4.1" }, { id: "gpt-4o" }] },
      ],
      default: { "google-vertex-anthropic": "claude-sonnet-4-6@default" },
      connected: ["google-vertex-anthropic"],
    })
  }),

  http.get("*/agent/provider/auth", () => {
    return HttpResponse.json({
      anthropic: [{ type: "api", label: "API Key", prompts: [{ type: "text", key: "key", message: "Anthropic API Key" }] }],
      openai: [{ type: "api", label: "API Key", prompts: [{ type: "text", key: "key", message: "OpenAI API Key" }] }],
    })
  }),

  http.post("*/agent/provider/:providerID/oauth/authorize", () => {
    return HttpResponse.json(null)
  }),

  // OpenCode agent: sessions
  http.post("*/agent/session", () => {
    return HttpResponse.json({ id: "ses_mock001" })
  }),

  http.post("*/agent/session/:sessionId/message", () => {
    return HttpResponse.json({
      info: {
        providerID: "mock",
        modelID: "mock-model",
        cost: 0,
        tokens: { input: 10, output: 20, reasoning: 0 },
        finish: "stop",
        id: "msg_mock001",
        sessionID: "ses_mock001",
      },
      parts: [
        { type: "text", text: "Hello! How can I help you build a task model?" },
      ],
    })
  }),

  // MLflow: Experiments search
  http.post("*/mlflow/api/2.0/mlflow/experiments/search", () => {
    return HttpResponse.json({
      experiments: [
        { experiment_id: "1", name: "amortized/sdg/test" },
        { experiment_id: "2", name: "amortized/training/test" },
      ],
    })
  }),

  // MLflow: Runs search (datasets)
  http.post("*/mlflow/api/2.0/mlflow/runs/search", () => {
    return HttpResponse.json(mockMlflowRunsSearch)
  }),

  // MLflow: Get run
  http.get("*/mlflow/api/2.0/mlflow/runs/get", () => {
    return HttpResponse.json({ run: mockMlflowRunsSearch.runs![0] })
  }),

  // MLflow: Metric history
  http.get("*/mlflow/api/2.0/mlflow/metrics/get-history", () => {
    return HttpResponse.json({
      metrics: [
        { key: "loss", value: 2.5, step: 1, timestamp: 1717200000000 },
        { key: "loss", value: 2.3, step: 2, timestamp: 1717200001000 },
      ],
    })
  }),

  // MLflow: Model Registry
  http.get("*/mlflow/api/2.0/mlflow/registered-models/search", () => {
    return HttpResponse.json(mockRegisteredModels)
  }),

  http.get("*/mlflow/api/2.0/mlflow/model-versions/search", () => {
    return HttpResponse.json(mockModelVersions)
  }),

  // MLflow: AI Gateway
  http.get("*/mlflow/api/2.0/mlflow/gateway/routes", () => {
    return HttpResponse.json({
      routes: [
        { name: "openai-gpt4o", route_type: "llm/v1/chat", model: { name: "gpt-4o", provider: "openai" } },
      ],
    })
  }),

  http.post("*/mlflow/api/2.0/mlflow/gateway/routes", () => {
    return HttpResponse.json(
      { name: "new-route", route_type: "llm/v1/chat", model: { name: "gpt-4o-mini", provider: "openai" } },
      { status: 201 },
    )
  }),

  http.delete("*/mlflow/api/2.0/mlflow/gateway/routes/:name", () => {
    return new HttpResponse(null, { status: 204 })
  }),
]
