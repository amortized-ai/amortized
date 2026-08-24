export interface OptionCard {
  title: string
  description: string
  value: string
}

export interface ProposedAction {
  action: string
  description: string
  params: Record<string, unknown>
  jobType?: "sdg" | "training"
  endpoint?: string
  config?: Record<string, unknown>
  parentJobId?: string
  recipe?: string
}

export interface ToolResult {
  name: string
  result: string
  collapsed: boolean
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: string
  toolResults: ToolResult[]
  proposedAction: ProposedAction | null
  optionCards: OptionCard[]
  selectedOptionValue?: string
  phase?: string
  streamStartedAt?: number
}

export type ChatState =
  | "idle"
  | "streaming"
  | "tool_call"
  | "action_pending"
  | "done"
  | "error"

export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
}

export interface PlanStep {
  label: string
  status: "pending" | "active" | "completed"
}

export type PlanPhase = "sdg" | "training" | "eval"

export interface PhasePlan {
  phase: PlanPhase
  label: string
  steps: PlanStep[]
}

export interface OpenCodePartText {
  type: "text"
  text: string
}

export interface OpenCodePartToolInvocation {
  type: "tool-invocation"
  toolName: string
  input: Record<string, unknown>
}

export interface OpenCodePartToolResult {
  type: "tool-result"
  result: unknown
}

export interface OpenCodePartStepStart {
  type: "step-start"
}

export interface OpenCodePartStepFinish {
  type: "step-finish"
}

export interface OpenCodePartReasoning {
  type: "reasoning"
  text: string
}

export interface OpenCodePartTool {
  type: "tool"
  tool: string
  callID: string
  state: string | Record<string, unknown>
  input?: Record<string, unknown>
  output?: unknown
}

export type OpenCodePart =
  | OpenCodePartText
  | OpenCodePartToolInvocation
  | OpenCodePartToolResult
  | OpenCodePartStepStart
  | OpenCodePartStepFinish
  | OpenCodePartReasoning
  | OpenCodePartTool

export interface OpenCodeResponse {
  info: {
    providerID: string
    modelID: string
    cost: number
    tokens: {
      input: number
      output: number
      reasoning: number
    }
    finish: string
    id: string
    sessionID: string
  }
  parts: OpenCodePart[]
}
