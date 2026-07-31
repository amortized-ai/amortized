export interface ChatModel {
  providerID: string
  modelID: string
  label: string
}

export interface ProviderInfo {
  label: string
  requiresApiKey: boolean
  models: ChatModel[]
}

export const PROVIDER_CATALOG: Record<string, ProviderInfo> = {
  "google-vertex-anthropic": {
    label: "Anthropic (Vertex AI)",
    requiresApiKey: false,
    models: [
      { providerID: "google-vertex-anthropic", modelID: "claude-opus-4-8@default", label: "Claude Opus 4.8" },
      { providerID: "google-vertex-anthropic", modelID: "claude-opus-4-6@default", label: "Claude Opus 4.6" },
      { providerID: "google-vertex-anthropic", modelID: "claude-sonnet-5@default", label: "Claude Sonnet 5" },
      { providerID: "google-vertex-anthropic", modelID: "claude-sonnet-4-6@default", label: "Claude Sonnet 4.6" },
      { providerID: "google-vertex-anthropic", modelID: "claude-haiku-4-5@20251001", label: "Claude Haiku 4.5" },
    ],
  },
  "anthropic": {
    label: "Anthropic (Direct)",
    requiresApiKey: true,
    models: [
      { providerID: "anthropic", modelID: "claude-opus-4-8-latest", label: "Claude Opus 4.8" },
      { providerID: "anthropic", modelID: "claude-opus-4-6-latest", label: "Claude Opus 4.6" },
      { providerID: "anthropic", modelID: "claude-sonnet-5-latest", label: "Claude Sonnet 5" },
      { providerID: "anthropic", modelID: "claude-sonnet-4-6-latest", label: "Claude Sonnet 4.6" },
      { providerID: "anthropic", modelID: "claude-haiku-4-5-20251001", label: "Claude Haiku 4.5" },
    ],
  },
  "openai": {
    label: "OpenAI",
    requiresApiKey: true,
    models: [
      { providerID: "openai", modelID: "gpt-4.1", label: "GPT-4.1" },
      { providerID: "openai", modelID: "gpt-4o", label: "GPT-4o" },
      { providerID: "openai", modelID: "gpt-4o-mini", label: "GPT-4o Mini" },
    ],
  },
}

const SEPARATOR = "::"

export function encodeModelSelection(providerID: string, modelID: string): string {
  return `${providerID}${SEPARATOR}${modelID}`
}

export function parseModelSelection(encoded: string): { providerID: string; modelID: string } {
  const idx = encoded.indexOf(SEPARATOR)
  if (idx === -1) return { providerID: "google-vertex-anthropic", modelID: encoded }
  return { providerID: encoded.slice(0, idx), modelID: encoded.slice(idx + SEPARATOR.length) }
}

export const DEFAULT_CHAT_MODEL_SELECTION = encodeModelSelection("google-vertex-anthropic", "claude-opus-4-8@default")
