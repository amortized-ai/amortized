// Baked opencode provider shim for BYOK MaaS (an OpenAI-dialect gateway, e.g. an internal MaaS).
//
// Why this exists: opencode routes a stock `@ai-sdk/openai` custom provider to the /responses API,
// but a MaaS endpoint is /v1/chat/completions-only; and `@ai-sdk/openai-compatible` always sends
// `max_tokens`, which reasoning-era OpenAI models (gpt-5/6) reject — they require
// `max_completion_tokens`, while the others (glm, gemini, deepseek, gpt-4o) need `max_tokens`.
//
// This shim wraps `@ai-sdk/openai` so `.languageModel(id)` delegates to `.chat(id)`: that forces
// /v1/chat/completions AND keeps @ai-sdk/openai's model-aware token-param selection (it version-parses
// the id, so custom names like gpt-5.6-luna / gpt-6-sol are handled). opencode calls the first
// `create*` export with the provider block's `options` ({ baseURL, apiKey }).
import { createOpenAI } from '@ai-sdk/openai';

export function createMaasChat(options) {
  const base = createOpenAI(options);
  const languageModel = (id) => base.chat(id);
  const provider = (id) => languageModel(id);
  provider.languageModel = languageModel;
  provider.chat = (id) => base.chat(id);
  provider.responses = (id) => base.responses(id);
  provider.textEmbeddingModel = (id) => base.textEmbeddingModel(id);
  return provider;
}
