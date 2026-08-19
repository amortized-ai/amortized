import { useSettingsStore } from "@/stores/settings-store"
import { getLogger } from "@/lib/logger"

const logger = getLogger("api-client")
import type {
  ConfigResponse,
  HealthResponse,
  Job,
  JobFilters,
  JobLogsResponse,
  MlflowGatewayRoute,
  MlflowMetricHistoryEntry,
  MlflowModelVersionsResponse,
  MlflowRegisteredModelsResponse,
  MlflowRun,
  MlflowRunsSearchResponse,
  PaginationParams,
  Recipe,
} from "@/types/api"

class ApiError extends Error {
  declare status: number
  declare statusText: string
  declare body: unknown

  constructor(status: number, statusText: string, body: unknown) {
    let detail: string | undefined
    if (typeof body === "object" && body !== null && "detail" in body) {
      detail = String((body as Record<string, unknown>).detail)
    } else if (typeof body === "string" && body.length > 0 && body.length < 200 && !body.includes("<html")) {
      detail = body
    }
    const friendly = detail ?? `API error: ${status} ${statusText}`
    super(friendly)
    this.name = "ApiError"
    this.status = status
    this.statusText = statusText
    this.body = body
  }
}

function getBaseUrl(): string {
  return import.meta.env.VITE_API_URL ?? ""
}

function getAuthHeaders(): Record<string, string> {
  const { apiKey } = useSettingsStore.getState()
  if (apiKey) {
    return { Authorization: `Bearer ${apiKey}` }
  }
  return {}
}

function friendlyServiceError(path: string, status: number): string | null {
  if (status === 413) return "File too large. Maximum upload size is 500 MB."
  if (status !== 502 && status !== 503) return null
  if (path.startsWith("/api/")) return "Cannot reach the backend server. Make sure it's running."
  if (path.startsWith("/mlflow/")) return "Cannot reach MLflow. Check the MLflow tracking server."
  if (path.startsWith("/agent/")) return "Cannot reach the agent service. Make sure OpenCode is running."
  return null
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const method = init?.method ?? "GET"
  const requestId = crypto.randomUUID()
  const start = performance.now()
  logger.debug("request start", { method, path, requestId })

  const headers: Record<string, string> = {
    "X-Request-ID": requestId,
    ...getAuthHeaders(),
    ...(init?.headers as Record<string, string> | undefined),
  }
  if (!(init?.body instanceof FormData)) {
    headers["Content-Type"] = "application/json"
  }

  let response: Response
  try {
    response = await fetch(`${getBaseUrl()}${path}`, {
      ...init,
      headers,
    })
  } catch {
    const friendly = friendlyServiceError(path, 502)
    throw new ApiError(0, "Network Error", friendly ?? "Network request failed. Check your connection.")
  }

  const duration = Math.round(performance.now() - start)

  if (!response.ok) {
    const raw = await response.text()
    let body: unknown = raw
    try {
      body = JSON.parse(raw)
    } catch { /* ignore parse errors */ }
    logger.error("request failed", { method, path, status: response.status, duration, requestId })
    const friendly = friendlyServiceError(path, response.status)
    if (friendly) {
      throw new ApiError(response.status, response.statusText, friendly)
    }
    throw new ApiError(response.status, response.statusText, body)
  }

  logger.info("request complete", { method, path, status: response.status, duration, requestId })

  const text = await response.text()
  if (!text) return undefined as T
  return JSON.parse(text) as T
}

function get<T>(path: string): Promise<T> {
  return request<T>(path)
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body != null ? JSON.stringify(body) : undefined,
  })
}

function put<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    body: body != null ? JSON.stringify(body) : undefined,
  })
}

function del<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "DELETE",
    body: body != null ? JSON.stringify(body) : undefined,
  })
}

function buildQuery(params: Record<string, string | undefined>): string {
  const entries = Object.entries(params).filter(
    (pair): pair is [string, string] => pair[1] != null,
  )
  if (entries.length === 0) return ""
  return `?${new URLSearchParams(entries).toString()}`
}

// --- Jobs ---

export function getJobs(filters?: JobFilters, pagination?: PaginationParams): Promise<Job[]> {
  logger.debug("getJobs", { filters, pagination })
  const query = buildQuery({
    type: filters?.type,
    status: filters?.status,
    page: pagination?.page?.toString(),
    per_page: pagination?.per_page?.toString(),
    sort: pagination?.sort,
    order: pagination?.order,
  })
  return get<Job[]>(`/api/v1/jobs${query}`)
}

export function getJob(id: string): Promise<Job> {
  logger.debug("getJob", { id })
  return get<Job>(`/api/v1/jobs/${id}`)
}

export function cancelJob(id: string): Promise<Job> {
  logger.info("cancelJob", { id })
  return del<Job>(`/api/v1/jobs/${id}`)
}

export function deleteJob(id: string): Promise<void> {
  return post<void>(`/api/v1/jobs/${id}/delete`)
}

export async function getJobLogs(id: string, tail = 2000): Promise<string[]> {
  logger.debug("getJobLogs", { id, tail })
  const resp = await get<JobLogsResponse>(`/api/v1/jobs/${id}/logs?tail=${tail}`)
  return resp.logs
}

// --- Recipes ---

export function getRecipes(): Promise<Recipe[]> {
  logger.debug("getRecipes")
  return get<Recipe[]>("/api/v1/recipes")
}

export function getRecipe(name: string): Promise<Recipe> {
  logger.debug("getRecipe", { name })
  return get<Recipe>(`/api/v1/recipes/${name}`)
}

export function saveRecipe(name: string, data: { type: string; description: string; config: Record<string, unknown> }): Promise<Recipe> {
  logger.info("saveRecipe", { name })
  return put<Recipe>(`/api/v1/recipes/${name}`, data)
}

export function deleteRecipe(name: string): Promise<void> {
  return del<void>(`/api/v1/recipes/${encodeURIComponent(name)}`)
}

export function submitRecipe(data: { recipe: string; overrides?: Record<string, unknown> }): Promise<Job> {
  logger.info("submitRecipe", { recipe: data.recipe })
  return post<Job>("/api/v1/jobs/recipe", data)
}

export function createJob(endpoint: string, body: Record<string, unknown>): Promise<Job> {
  logger.info("createJob", { endpoint })
  return post<Job>(endpoint, body)
}

// --- Agent Chat (OpenCode) ---

import type { OpenCodeResponse } from "@/features/chat/types"
import { useChatStore } from "@/stores/chat-store"

let activeConversationId: string | null = null

async function getOrCreateSession(conversationId: string): Promise<string> {
  const existing = useChatStore.getState().getSessionId(conversationId)
  if (existing) return existing

  logger.info("creating OpenCode session", { conversationId })
  const resp = await fetch(`${getBaseUrl()}/agent/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  })
  if (!resp.ok) {
    const friendly = friendlyServiceError("/agent/", resp.status)
    throw new ApiError(resp.status, resp.statusText, friendly)
  }
  const data = await resp.json()
  const sessionId = data.id as string
  useChatStore.getState().setSessionId(conversationId, sessionId)
  logger.info("OpenCode session created", { conversationId, sessionId })
  return sessionId
}

export function setActiveConversation(conversationId: string | null): void {
  activeConversationId = conversationId
}

export function clearConversationSession(conversationId: string): void {
  useChatStore.getState().clearSessionId(conversationId)
}

export function resetOpenCodeSession(): void {
  if (activeConversationId) {
    useChatStore.getState().clearSessionId(activeConversationId)
  }
}

const MAX_RETRIES = 2

export async function sendOpenCodeMessage(conversationId: string, text: string, modelSelection?: string): Promise<OpenCodeResponse> {
  if (!conversationId) {
    throw new ApiError(400, "No active conversation", null)
  }
  let sessionId = await getOrCreateSession(conversationId)
  logger.info("sendOpenCodeMessage", { conversationId, sessionId, modelSelection })
  const body: Record<string, unknown> = {
    agent: "morty",
    parts: [{ type: "text", text }],
  }
  if (modelSelection) {
    const { parseModelSelection } = await import("@/features/chat/models")
    body.model = parseModelSelection(modelSelection)
  }

  let lastError: Error | null = null
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    let resp: Response
    try {
      resp = await fetch(`${getBaseUrl()}/agent/session/${sessionId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      })
    } catch (err) {
      lastError = err instanceof Error ? err : new Error(String(err))
      logger.warn("network error, resetting session", { attempt, error: lastError.message })
      useChatStore.getState().clearSessionId(conversationId)
      sessionId = await getOrCreateSession(conversationId)
      continue
    }
    if (resp.ok) {
      return resp.json()
    }
    lastError = new ApiError(resp.status, resp.statusText, null)
    if ((resp.status === 404 || resp.status === 500) && attempt < MAX_RETRIES) {
      logger.warn("session error, creating new session with context replay", { conversationId, sessionId, status: resp.status })
      useChatStore.getState().clearSessionId(conversationId)
      useChatStore.getState().setSessionStatus(conversationId, "reconnecting")
      sessionId = await getOrCreateSession(conversationId)

      const messages = useChatStore.getState().getConversationMessages(conversationId)
      if (messages.length > 0) {
        const { summarizeConversation } = await import("@/lib/context-summarizer")
        const summary = summarizeConversation(messages)
        if (summary) {
          try {
            await fetch(`${getBaseUrl()}/agent/session/${sessionId}/message`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ agent: "morty", parts: [{ type: "text", text: summary }] }),
            })
            logger.info("context replayed to new session", { conversationId, sessionId })
          } catch {
            logger.warn("failed to replay context", { conversationId })
          }
        }
        useChatStore.getState().setSessionStatus(conversationId, "restored")
      }
      continue
    }
    if (resp.status === 502 && attempt < MAX_RETRIES) {
      logger.warn("transient error, resetting session", {
        sessionId, status: resp.status, attempt: attempt + 1,
      })
      useChatStore.getState().clearSessionId(conversationId)
      sessionId = await getOrCreateSession(conversationId)
      continue
    }
    break
  }
  useChatStore.getState().clearSessionId(conversationId)
  if (lastError instanceof ApiError && (lastError.status === 502 || lastError.status === 503)) {
    throw new ApiError(lastError.status, lastError.statusText, "Cannot reach Morty. Make sure the agent service is running.")
  }
  if (lastError instanceof TypeError || (lastError instanceof Error && lastError.message.includes("fetch"))) {
    throw new ApiError(0, "Network Error", "Cannot reach Morty. Make sure the agent service is running.")
  }
  throw lastError!
}

export async function sendOpenCodeMessageWithContext(
  conversationId: string,
  contextSummary: string,
  userMessage: string,
  modelSelection?: string,
): Promise<OpenCodeResponse> {
  await sendOpenCodeMessage(conversationId, contextSummary, modelSelection)
  return sendOpenCodeMessage(conversationId, userMessage, modelSelection)
}

export async function fetchSessionMessages(
  conversationId: string,
): Promise<OpenCodeResponse[]> {
  const sessionId = useChatStore.getState().getSessionId(conversationId)
  if (!sessionId) return []
  try {
    const resp = await fetch(`${getBaseUrl()}/agent/session/${sessionId}/message`)
    if (!resp.ok) {
      logger.warn("fetchSessionMessages failed", { sessionId, status: resp.status })
      return []
    }
    const data = await resp.json()
    if (Array.isArray(data)) return data
    if (data && typeof data === "object" && "parts" in data) return [data]
    return []
  } catch (err) {
    logger.warn("fetchSessionMessages error", { sessionId, error: err instanceof Error ? err.message : String(err) })
    return []
  }
}

export async function fetchPendingMessages(
  conversationId: string,
): Promise<OpenCodeResponse[]> {
  const sessionId = useChatStore.getState().getSessionId(conversationId)
  if (!sessionId) return []
  try {
    const resp = await fetch(`${getBaseUrl()}/agent/session/${sessionId}/pending`)
    if (!resp.ok) return []
    const data = await resp.json()
    return data.messages ?? []
  } catch {
    return []
  }
}

export async function generateChatTitle(message: string): Promise<string> {
  const resp = await fetch(`${getBaseUrl()}/agent/title`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  })
  if (!resp.ok) return message.slice(0, 40) + (message.length > 40 ? "..." : "")
  const data = await resp.json()
  return data.title || message.slice(0, 40)
}

// --- System ---

export function getHealth(): Promise<HealthResponse> {
  logger.debug("getHealth")
  return get<HealthResponse>("/api/v1/health")
}

export function getConfig(): Promise<ConfigResponse> {
  logger.debug("getConfig")
  return get<ConfigResponse>("/api/v1/config")
}

// --- MLflow: Experiments ---

export function searchMlflowExperiments(): Promise<{
  experiments: { experiment_id: string; name: string }[]
}> {
  return post<{ experiments: { experiment_id: string; name: string }[] }>(
    "/mlflow/api/2.0/mlflow/experiments/search",
    { max_results: 200 },
  )
}

// --- MLflow: Runs ---

export function searchMlflowRuns(body: {
  experiment_ids?: string[]
  filter_string?: string
  max_results?: number
  order_by?: string[]
  page_token?: string
}): Promise<MlflowRunsSearchResponse> {
  return post<MlflowRunsSearchResponse>("/mlflow/api/2.0/mlflow/runs/search", body)
}

export function deleteDataset(runId: string): Promise<void> {
  return post<void>("/mlflow/api/2.0/mlflow/runs/delete", { run_id: runId })
}

export function getMlflowRun(runId: string): Promise<{ run: MlflowRun }> {
  return get<{ run: MlflowRun }>(`/mlflow/api/2.0/mlflow/runs/get${buildQuery({ run_id: runId })}`)
}

export function getMlflowMetricHistory(
  runId: string,
  metricKey: string,
): Promise<{ metrics: MlflowMetricHistoryEntry[] }> {
  return get<{ metrics: MlflowMetricHistoryEntry[] }>(
    `/mlflow/api/2.0/mlflow/metrics/get-history${buildQuery({ run_id: runId, metric_key: metricKey })}`,
  )
}

// --- MLflow: Model Registry ---

export function searchMlflowRegisteredModels(
  filter?: string,
  maxResults?: number,
): Promise<MlflowRegisteredModelsResponse> {
  const query = buildQuery({ filter, max_results: maxResults?.toString() })
  return get<MlflowRegisteredModelsResponse>(`/mlflow/api/2.0/mlflow/registered-models/search${query}`)
}

export function searchMlflowModelVersions(filter: string): Promise<MlflowModelVersionsResponse> {
  const query = buildQuery({ filter })
  return get<MlflowModelVersionsResponse>(`/mlflow/api/2.0/mlflow/model-versions/search${query}`)
}

export function deleteMlflowRegisteredModel(name: string): Promise<void> {
  return del<void>("/mlflow/api/2.0/mlflow/registered-models/delete", { name })
}

export function setMlflowRunTag(runId: string, key: string, value: string): Promise<void> {
  return post<void>("/mlflow/api/2.0/mlflow/runs/set-tag", { run_id: runId, key, value })
}

export function updateMlflowRunName(runId: string, name: string): Promise<void> {
  return post<void>("/mlflow/api/2.0/mlflow/runs/update", { run_id: runId, run_name: name })
}

export function renameMlflowRegisteredModel(name: string, newName: string): Promise<void> {
  return post<void>("/mlflow/api/2.0/mlflow/registered-models/rename", { name, new_name: newName })
}

export function setMlflowRegisteredModelTag(name: string, key: string, value: string): Promise<void> {
  return post<void>("/mlflow/api/2.0/mlflow/registered-models/set-tag", { name, key, value })
}

// --- MLflow: AI Gateway (v3 endpoints API) ---

import type { MlflowGatewayEndpoint } from "@/types/api"

export async function getMlflowGatewayRoutes(): Promise<{ routes: MlflowGatewayRoute[] }> {
  const data = await get<{ endpoints: MlflowGatewayEndpoint[] }>(
    "/mlflow/api/3.0/mlflow/gateway/endpoints/list"
  )
  const routes: MlflowGatewayRoute[] = (data.endpoints ?? []).map((ep) => {
    const primary = ep.model_mappings?.find((m) => m.linkage_type === "PRIMARY")
    return {
      name: ep.name,
      route_type: "llm/v1/chat",
      model: {
        name: primary?.model_definition?.model_name ?? "",
        provider: primary?.model_definition?.provider ?? "",
      },
      endpoint_id: ep.endpoint_id,
    }
  })
  return { routes }
}

// --- Datasets ---

export interface DatasetListItem {
  run_id: string
  name: string
  topic: string
  source: string
  samples: string
  teacher_model: string
  job_id: string
  experiment_id: string
  created_at: number
}

export function listDatasets(search = ""): Promise<DatasetListItem[]> {
  const query = search ? buildQuery({ search }) : ""
  return get<DatasetListItem[]>(`/api/v1/datasets${query}`)
}

export function uploadDataset(file: File): Promise<Job> {
  const formData = new FormData()
  formData.append("file", file)
  return request<Job>("/api/v1/datasets/upload", {
    method: "POST",
    body: formData,
  })
}

// --- Documents ---

import type { DocumentChunksResponse, DocumentRecord, DocumentUploadAccepted, DocumentUploadResponse } from "@/types/api"

export function getDocuments(): Promise<DocumentRecord[]> {
  return get<DocumentRecord[]>("/api/v1/documents")
}

export function getDocumentContent(id: string): Promise<DocumentUploadResponse> {
  return get<DocumentUploadResponse>(`/api/v1/documents/${id}/content`)
}

export function getDocumentChunks(id: string): Promise<DocumentChunksResponse> {
  return get<DocumentChunksResponse>(`/api/v1/documents/${id}/chunks`)
}

export function deleteDocument(id: string): Promise<void> {
  return del<void>(`/api/v1/documents/${id}`)
}

export interface ChunkOptions {
  output_format?: string
  chunker_type?: string
  chunk_size?: number
  chunk_overlap?: number
}

export async function uploadDocument(
  file: File,
  options: ChunkOptions = {},
): Promise<DocumentUploadAccepted> {
  const formData = new FormData()
  formData.append("file", file)
  const params = new URLSearchParams()
  if (options.output_format) params.set("output_format", options.output_format)
  if (options.chunker_type) params.set("chunker_type", options.chunker_type)
  if (options.chunk_size != null) params.set("chunk_size", String(options.chunk_size))
  if (options.chunk_overlap != null) params.set("chunk_overlap", String(options.chunk_overlap))
  const qs = params.toString()
  return request<DocumentUploadAccepted>(`/api/v1/documents/convert${qs ? `?${qs}` : ""}`, {
    method: "POST",
    body: formData,
  })
}

export function convertDocumentUrl(
  url: string,
  options: ChunkOptions = {},
): Promise<DocumentUploadAccepted> {
  return post<DocumentUploadAccepted>("/api/v1/documents/convert/url", {
    url,
    options: {
      output_format: options.output_format ?? "md",
      chunker_type: options.chunker_type ?? "sentence",
      chunk_size: options.chunk_size ?? 2048,
      chunk_overlap: options.chunk_overlap ?? 200,
    },
  })
}

// --- MLflow: Artifacts ---

/**
 * Fetch raw artifact content (e.g. a JSONL file) from the MLflow artifacts API.
 * Returns the raw text — caller is responsible for parsing.
 */
export async function getMlflowArtifactContent(
  experimentId: string,
  runId: string,
  artifactPath: string,
): Promise<string> {
  const url = `/api/v1/artifacts/${encodeURIComponent(experimentId)}/${encodeURIComponent(runId)}/${artifactPath}`
  logger.debug("getMlflowArtifactContent", { experimentId, runId, artifactPath })

  const response = await fetch(`${getBaseUrl()}${url}`, {
    headers: {
      "X-Request-ID": crypto.randomUUID(),
      ...getAuthHeaders(),
    },
  })

  if (!response.ok) {
    const raw = await response.text()
    let body: unknown = raw
    try { body = JSON.parse(raw) } catch { /* ignore parse errors */ }
    throw new ApiError(response.status, response.statusText, body)
  }

  return response.text()
}

export async function listArtifacts(
  experimentId: string,
  runId: string,
  path = "",
): Promise<{ files: { path: string; file_size: number }[] }> {
  const query = path ? `?path=${encodeURIComponent(path)}` : ""
  return get<{ files: { path: string; file_size: number }[] }>(
    `/api/v1/artifacts/${encodeURIComponent(experimentId)}/${encodeURIComponent(runId)}${query}`,
  )
}

export async function getArtifactJson(
  experimentId: string,
  runId: string,
  artifactPath: string,
): Promise<Record<string, unknown>[]> {
  const url = `/api/v1/artifacts/${encodeURIComponent(experimentId)}/${encodeURIComponent(runId)}/${artifactPath}`
  const response = await fetch(`${getBaseUrl()}${url}`, {
    headers: {
      "X-Request-ID": crypto.randomUUID(),
      ...getAuthHeaders(),
    },
  })
  if (!response.ok) {
    throw new ApiError(response.status, response.statusText, null)
  }
  return response.json()
}

export { ApiError }
