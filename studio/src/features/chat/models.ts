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
  "maas": {
    label: "MaaS Gateway (Gemini)",
    requiresApiKey: false,
    models: [
      { providerID: "maas", modelID: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
      { providerID: "maas", modelID: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
      { providerID: "maas", modelID: "gemini-3.5-flash", label: "Gemini 3.5 Flash" },
    ],
  },
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
      { providerID: "openai", modelID: "gpt-5.6-sol", label: "GPT-5.6 Sol" },
      { providerID: "openai", modelID: "gpt-5.6-terra", label: "GPT-5.6 Terra" },
      { providerID: "openai", modelID: "gpt-5.6-luna", label: "GPT-5.6 Luna" },
      { providerID: "openai", modelID: "gpt-5.5-2026-04-23", label: "GPT-5.5 (2026-04-23)" },
      { providerID: "openai", modelID: "gpt-5.4-2026-03-05", label: "GPT-5.4 (2026-03-05)" },
    ],
  },
}

const SEPARATOR = "::"

export function encodeModelSelection(providerID: string, modelID: string): string {
  return `${providerID}${SEPARATOR}${modelID}`
}

export function parseModelSelection(encoded: string): { providerID: string; modelID: string } {
  const idx = encoded.indexOf(SEPARATOR)
  if (idx === -1) return { providerID: "maas", modelID: encoded }
  return { providerID: encoded.slice(0, idx), modelID: encoded.slice(idx + SEPARATOR.length) }
}

export const DEFAULT_CHAT_MODEL_SELECTION = encodeModelSelection("maas", "gemini-2.5-flash")
