// Baked opencode provider shim for BYOK MaaS (any OpenAI-compatible endpoint, e.g. an internal MaaS).
//
// Routing: EVERY model goes through /v1/chat/completions via @ai-sdk/openai's `.chat(id)`.
// chat-completions is the UNIVERSAL OpenAI-compatible dialect — every such endpoint implements it,
// including chat-only ones (unlike the Responses API, which many compatible servers and EnMaaS's
// gemini do not have). @ai-sdk/openai's `.chat()` is model-aware: it sends `max_completion_tokens`
// for reasoning ids (gpt-5.x / gpt-6.x / o-series) and `max_tokens` for the rest — which fixes the
// gpt-5/6 "max_tokens is not supported" rejection that the stock @ai-sdk/openai-compatible provider
// caused (that one always sends `max_tokens` and is not model-aware).
//
// Why a shim at all: opencode routes a stock @ai-sdk/openai CUSTOM provider to /v1/responses for ALL
// ids (which 404s on chat-only endpoints and on EnMaaS gemini), and @ai-sdk/openai-compatible is
// chat-only but NOT model-aware. The shim is the one small place to force chat-completions WITH
// model-aware params.
//
// Known limitations of the chat-completions path (both match how `main` already behaves, and both
// are fixed natively by the opencode-v2 migration — see issue #570):
//   - An id @ai-sdk/openai can't classify by prefix (e.g. `codex-auto-review`, which is neither
//     `gpt-*` nor `o<N>`) gets `max_tokens`; if it is a reasoning model whose endpoint enforces the
//     OpenAI param rule, that single id fails. All real gpt-5.x / gpt-6.x ids ARE classified right.
//   - A model that returns `reasoning_content` (e.g. deepseek thinking mode) does not have it
//     replayed across tool calls, so deepseek thinking + tools can fail mid-turn.
//
// opencode (file:// provider) calls the first create* export, then provider.languageModel(id), with
// options { baseURL, apiKey }.
import { createOpenAI } from '@ai-sdk/openai';

export function createMaasChat(options) {
  const base = createOpenAI(options);
  const languageModel = (id) => base.chat(id);
  const provider = (id) => languageModel(id);
  provider.languageModel = languageModel;
  return provider;
}
