import type { ChatMessage } from "../types"

export const FRIENDLY_MODEL_NAMES: Record<string, string> = {
  "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
  "claude-haiku-4-5@20251001": "Claude Haiku 4.5",
  "claude-sonnet-4-20250514": "Claude Sonnet 4",
  "claude-sonnet-4@20250514": "Claude Sonnet 4",
  "gpt-4o": "GPT-4o",
  "gpt-4o-mini": "GPT-4o Mini",
  "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
}

export function friendlyModelName(nameOrId: string): string {
  const exact = FRIENDLY_MODEL_NAMES[nameOrId]
  if (exact) return exact
  const withoutProvider = nameOrId.replace(/^(openai|anthropic|vertex_ai)\//, "")
  const fromStripped = FRIENDLY_MODEL_NAMES[withoutProvider]
  if (fromStripped) return fromStripped
  if (/^[A-Z]/.test(nameOrId) && !nameOrId.includes("/")) return nameOrId
  return withoutProvider
    .replace(/-/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase())
}

export type WorkflowPhase = "sdg" | "training"

export function parsePhaseTag(phase: string): { phase: WorkflowPhase; step: string } | null {
  const parts = phase.split(":")
  if (parts.length !== 2) return null
  const [p, step] = parts as [string, string]
  if (p !== "sdg" && p !== "training") return null
  return { phase: p, step }
}

export function getLatestPhase(messages: ChatMessage[]): string | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const phase = messages[i]!.phase
    if (phase) return phase
  }
  return null
}
