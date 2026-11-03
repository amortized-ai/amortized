// Baked opencode provider shim for BYOK MaaS (an OpenAI-dialect gateway, e.g. an internal MaaS).
//
// Routing: EVERYTHING goes through the modern Responses API (/v1/responses) via @ai-sdk/openai,
// EXCEPT gemini, which goes through chat-completions. Responses is the forward path — it sidesteps
// the max_tokens-vs-max_completion_tokens problem entirely (it uses max_output_tokens) and handles
// reasoning natively. gemini is the one EnMaaS model with no /v1/responses endpoint (404), so it
// must use chat-completions (@ai-sdk/openai's `.chat`).
//
// Why a shim is needed at all: opencode routes a stock @ai-sdk/openai custom provider to Responses
// for ALL ids with no per-model override (so gemini would 404), and @ai-sdk/openai-compatible only
// speaks chat-completions + always sends max_tokens (which gpt-5/6 reject). The shim is the one
// small place to say "gemini -> chat, everything else -> responses". opencode (file:// provider)
// calls the first `create*` export, then provider.languageModel(id), with options { baseURL, apiKey }.
//
// REVISIT (post-0.2.2): if gemini gains /v1/responses on EnMaaS, OR if supporting gemini via maas is
// dropped, delete this shim entirely — the maas provider can then just be npm:"@ai-sdk/openai".
// gemini is currently undocumented/unsupported by EnMaaS (absent from the docs; catalog-listed +
// chat-completions-only). Verified 2026-11-02: /v1/responses works for gpt-5/6, glm, deepseek;
// gemini-3.x -> HTTP 404.
import { createOpenAI } from '@ai-sdk/openai';

export function createMaasChat(options) {
  const base = createOpenAI(options);
  const pick = (id) => (/gemini/i.test(id) ? base.chat(id) : base.responses(id));
  const languageModel = (id) => pick(id);
  const provider = (id) => languageModel(id);
  provider.languageModel = languageModel;
  return provider;
}
