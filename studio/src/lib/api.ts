// --- Types matching FastAPI Pydantic models ---

export type JobType = "training" | "sdg";
export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface TrainingJobConfig {
  model_path: string;
  data_path: string;
  ckpt_output_dir: string;
  learning_rate?: number | null;
  num_epochs?: number | null;
  lora_r?: number | null;
  lora_alpha?: number | null;
  load_in_4bit?: boolean | null;
  micro_batch_size?: number | null;
  max_seq_len?: number | null;
}

export interface SDGJobConfig {
  model: string;
  num_samples?: number;
  max_concurrent?: number;
  temperature?: number;
  max_tokens?: number | null;
  top_p?: number | null;
  seed?: number | null;
  num_retries?: number | null;
  api_base?: string | null;
  api_key?: string | null;
  strategy_params?: Record<string, unknown> | null;
  input_data?: Record<string, unknown>[] | null;
  input_documents?: Record<string, unknown>[] | null;
}

export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  config: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  error?: string | null;
  output_dir?: string | null;
}

export interface Artifact {
  id: string;
  job_id: string;
  artifact_type: string;
  path: string;
  size: number;
  created_at: string;
}

export interface TrainingMetric {
  step: number;
  loss: number;
  epoch?: number | null;
  learning_rate?: number | null;
  max_steps?: number | null;
}

export interface MemoryEstimate {
  model_path: string;
  lora_r: number;
  batch_size: number;
  max_seq_len: number;
  estimated_vram_gb: number;
  load_in_4bit: boolean;
}

export interface SDGFlow {
  name: string;
  description: string;
  supports_multi_turn: boolean;
  config_schema?: Record<string, unknown>;
}

// --- API base ---

const API_BASE = "/api/v1";

// Direct runtime URL for SSE streaming (bypasses Next.js proxy which buffers SSE)
const RUNTIME_URL = process.env.NEXT_PUBLIC_RUNTIME_URL || "http://localhost:8000";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// --- Fetch functions ---

export async function listJobs(status?: JobStatus, type?: JobType): Promise<Job[]> {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (type) params.set("type", type);
  const qs = params.toString();
  return apiFetch<Job[]>(`/jobs${qs ? `?${qs}` : ""}`);
}

export async function getJob(id: string): Promise<Job> {
  return apiFetch<Job>(`/jobs/${id}`);
}

export async function getJobMetrics(id: string): Promise<TrainingMetric[]> {
  return apiFetch<TrainingMetric[]>(`/jobs/${id}/metrics`);
}

export async function getJobArtifacts(id: string): Promise<Artifact[]> {
  return apiFetch<Artifact[]>(`/jobs/${id}/artifacts`);
}

export function getArtifactDownloadUrl(jobId: string, artifactId: string): string {
  return `${RUNTIME_URL}/api/v1/jobs/${jobId}/artifacts/${artifactId}/download`;
}

export async function createTrainingJob(config: TrainingJobConfig): Promise<Job> {
  return apiFetch<Job>("/jobs/training", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function createSDGJob(config: SDGJobConfig): Promise<Job> {
  return apiFetch<Job>("/jobs/sdg", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function cancelJob(id: string): Promise<Job> {
  return apiFetch<Job>(`/jobs/${id}`, { method: "DELETE" });
}

export async function listFlows(): Promise<SDGFlow[]> {
  return apiFetch<SDGFlow[]>("/flows");
}

export async function estimateMemory(config: {
  model_path: string;
  lora_r?: number;
  batch_size?: number;
  max_seq_len?: number;
  load_in_4bit?: boolean;
}): Promise<MemoryEstimate> {
  return apiFetch<MemoryEstimate>("/estimate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

// --- Agent / Chat types ---

export interface SuggestedAction {
  type: string;
  config: Record<string, unknown>;
  label: string;
}

export interface ChatResponse {
  conversation_id: string;
  message: string;
  suggested_action?: SuggestedAction | null;
  context?: Record<string, unknown>;
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string | { message: string; suggested_action?: SuggestedAction | null };
  created_at: string;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: ChatMessage[];
}

// --- Agent API functions ---

export async function sendChatMessage(
  message: string,
  conversationId?: string
): Promise<ChatResponse> {
  return apiFetch<ChatResponse>("/agent/chat", {
    method: "POST",
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  });
}

export interface ThinkingEvent {
  tool: string;
}

export interface ToolResultEvent {
  tool: string;
  summary: string;
}

export interface ActionEvent {
  type: string;
  config: Record<string, unknown>;
  label: string;
}

export interface StreamCallbacks {
  onDelta: (text: string) => void;
  onDone: (fullText: string) => void;
  onMetadata: (data: { conversation_id: string }) => void;
  onError: (error: string) => void;
  onThinking?: (data: ThinkingEvent) => void;
  onToolResult?: (data: ToolResultEvent) => void;
  onAction?: (data: ActionEvent) => void;
}

export async function streamChatMessage(
  message: string,
  conversationId: string | undefined,
  onDelta: (text: string) => void,
  onDone: (fullText: string) => void,
  onMetadata: (data: { conversation_id: string }) => void,
  onError: (error: string) => void,
  extra?: {
    onThinking?: (data: ThinkingEvent) => void;
    onToolResult?: (data: ToolResultEvent) => void;
    onAction?: (data: ActionEvent) => void;
  }
): Promise<void> {
  const res = await fetch(`${RUNTIME_URL}/api/v1/agent/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    onError(`API error ${res.status}: ${text}`);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    let eventType = "";
    for (const line of lines) {
      if (line.startsWith("event:")) {
        eventType = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        const data = line.slice(5).trim();
        if (!data) continue;
        try {
          const parsed = JSON.parse(data);
          switch (eventType) {
            case "metadata":
              onMetadata(parsed);
              break;
            case "delta":
              onDelta(parsed.text);
              break;
            case "thinking":
              extra?.onThinking?.(parsed);
              break;
            case "tool_result":
              extra?.onToolResult?.(parsed);
              break;
            case "action":
              extra?.onAction?.(parsed);
              break;
            case "done":
              onDone(parsed.full_text);
              break;
            case "error":
              onError(parsed.error);
              break;
          }
        } catch {
          // skip malformed SSE data
        }
      }
    }
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  return apiFetch<ConversationSummary[]>("/agent/conversations");
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  return apiFetch<ConversationDetail>(`/agent/conversations/${id}`);
}
