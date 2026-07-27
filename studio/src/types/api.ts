export type JobType = "training" | "sdg" | "eval"

export type JobStatus =
  | "queued"
  | "provisioning"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"

export interface Job {
  id: string
  type: JobType
  status: JobStatus
  config: Record<string, unknown>
  metadata: Record<string, unknown>
  recipe: string
  user_id: string | null
  k8s_job_name: string | null
  k8s_namespace: string | null
  mlflow_run_id: string | null
  mlflow_experiment: string | null
  parent_job_id: string | null
  error: string | null
  created_at: string
  started_at: string | null
  completed_at: string | null
}

export interface ComputeSpec {
  backend?: string
  gpus?: number
  gpu_type?: string
}

export interface JobRequest {
  type: string
  config: Record<string, unknown>
  metadata?: Record<string, unknown>
  recipe?: string
  parent_job_id?: string
  compute?: ComputeSpec
  dry_run?: boolean
}

export interface Recipe {
  name: string
  type: string
  description: string
  version: string
  schema: Record<string, unknown>
  defaults: Record<string, unknown>
}

export interface TrainingMetric {
  step: number
  loss: number
  learning_rate: number | null
  grad_norm: number | null
  epoch: number | null
  max_steps?: number | null
}

// --- Chat (untouched) ---

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

export interface ChatRequest {
  conversation_id?: string
  message: string
}

export interface SuggestedAction {
  type: string
  config: Record<string, unknown>
  label: string
}

export interface ChatResponse {
  conversation_id: string
  message: ChatMessage
  suggested_action?: SuggestedAction
  context?: Record<string, unknown>
}

export interface SSEEvent {
  type: "metadata" | "thinking" | "tool_result" | "delta" | "action" | "options" | "done" | "error"
  data: Record<string, unknown>
}

export interface JobLogsResponse {
  job_id: string
  logs: string[]
  message: string
}

// --- GPU ---

export interface GpuNodeMetrics {
  index: number
  name: string
  utilization_pct: number
  memory_used_mb: number
  memory_total_mb: number
  temperature_c: number
}

export interface GpuUtilizationResponse {
  nodes: GpuNodeMetrics[]
}

// --- System ---

export interface HealthResponse {
  status: "ok"
  timestamp: string
  version?: string
  gpu?: {
    available?: boolean
    count?: number
    devices?: string[]
    note?: string
  }
}

export interface ConfigResponse {
  mlflow_tracking_uri: string
  mlflow_gateway_uri: string | null
  default_compute_backend: string
  compute_namespace: string
  image_registry: string
  available_backends: string[]
  version: string
}

// --- MLflow: Experiment Tracking ---

export interface MlflowRunInfo {
  run_id: string
  experiment_id: string
  status: string
  start_time: number
  end_time: number | null
  artifact_uri: string
  run_name: string | null
}

export interface MlflowRunData {
  metrics: Array<{ key: string; value: number; step: number; timestamp: number }>
  params: Array<{ key: string; value: string }>
  tags: Array<{ key: string; value: string }>
}

export interface MlflowRun {
  info: MlflowRunInfo
  data: MlflowRunData
}

export interface MlflowRunsSearchResponse {
  runs?: MlflowRun[]
  next_page_token?: string
}

export interface MlflowMetricHistoryEntry {
  key: string
  value: number
  step: number
  timestamp: number
}

// --- MLflow: Model Registry ---

export interface MlflowModelVersion {
  name: string
  version: string
  creation_timestamp: number
  last_updated_timestamp: number
  current_stage: string
  source: string
  run_id: string
  status: string
  tags?: Array<{ key: string; value: string }>
  aliases?: string[]
}

export interface MlflowRegisteredModel {
  name: string
  creation_timestamp: number
  last_updated_timestamp: number
  description: string
  latest_versions?: MlflowModelVersion[]
  tags?: Array<{ key: string; value: string }>
  aliases?: Record<string, string>
}

export interface MlflowRegisteredModelsResponse {
  registered_models?: MlflowRegisteredModel[]
  next_page_token?: string
}

export interface MlflowModelVersionsResponse {
  model_versions?: MlflowModelVersion[]
  next_page_token?: string
}

// --- MLflow: AI Gateway ---

export interface MlflowGatewayRoute {
  name: string
  route_type: string
  model: { name: string; provider: string }
  endpoint_id?: string
}

export interface MlflowGatewayEndpoint {
  endpoint_id: string
  name: string
  created_at: number
  last_updated_at: number
  model_mappings: {
    model_definition: {
      provider: string
      model_name: string
    }
    linkage_type: string
  }[]
}

export interface MlflowGatewayRouteCreate {
  name: string
  route_type: string
  model: { name: string; provider: string }
  secret_name: string
}

// --- MLflow: AI Gateway Connections ---

export interface MlflowGatewayConnection {
  secret_id: string
  secret_name: string
  provider: string | null
  created_at: number
  last_updated_at: number | null
  masked_values: Record<string, string> | null
}

export interface MlflowGatewayConnectionCreate {
  name: string
  provider: string
  apiKey: string
}

// --- Derived view types ---

export interface DatasetRecord {
  run_id: string
  name: string
  run_name: string | null
  experiment_id: string
  artifact_uri: string
  created_at: number
  metrics: Record<string, number>
  params: Record<string, string>
  tags: Record<string, string>
}

export interface DatasetSample {
  index: number
  messages: Array<{ role: string; content: string }>
  metadata: Record<string, unknown>
}

export interface ModelRecord {
  name: string
  version: string
  run_id: string
  source: string
  created_at: number
  description: string
  aliases: string[]
  tags: Record<string, string>
}

// --- Documents ---

export interface DocumentRecord {
  document_id: string
  mlflow_run_id: string | null
  filename: string
  format: string
  created_at: string | null
}

export interface DocumentUploadResponse {
  document_id: string
  mlflow_run_id: string | null
  filename: string
  content: string
  format: string
  processing_time: number
  status: string
  warnings: string[]
}

// --- Query helpers ---

export interface JobFilters {
  type?: JobType
  status?: JobStatus
}

export interface PaginationParams {
  page?: number
  per_page?: number
  sort?: string
  order?: "asc" | "desc"
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  per_page: number
}
