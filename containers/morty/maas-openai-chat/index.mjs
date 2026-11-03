import { createOpenAICompatible } from '@ai-sdk/openai-compatible';

// opencode's `maas` provider shim: OpenAI-compatible chat-completions with a token-param rename.
export function createMaasChat(options) {
  return createOpenAICompatible({ ...options, name: 'maas', includeUsage: true, transformRequestBody });
}

// Reasoning models require max_completion_tokens; others accept it, so rename unconditionally.
export function transformRequestBody({ max_tokens, ...body }) {
  if (max_tokens == null) return { ...body };
  return { ...body, max_completion_tokens: body.max_completion_tokens ?? max_tokens };
}
