// Spots credentials pasted into the chat box, so the message is held back before it reaches the
// chat model and the chat history. Kept tight: a miss is no worse than before, and a false
// positive costs one click ("Send anyway").

// Known key formats, matched anywhere.
const KEY_PATTERNS: RegExp[] = [
  /\bsk-[A-Za-z0-9_-]{16,}/g, // OpenAI, Anthropic (sk-ant-), LiteLLM-style proxies
  /\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g, // JWTs (MaaS / service-account tokens)
  /\bhf_[A-Za-z0-9]{30,}/g, // Hugging Face
  /\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}/g, // GitHub tokens
  /\bgithub_pat_[A-Za-z0-9_]{30,}/g, // GitHub fine-grained tokens
  /\bAIza[0-9A-Za-z_-]{35}/g, // Google API keys
  /\bAKIA[0-9A-Z]{16}\b/g, // AWS access key ids
  /\bxox[abprs]-[A-Za-z0-9-]{10,}/g, // Slack tokens
]

// A long token right after a word like "key", "token" or "Bearer" ("api key: abc…", "token=…").
const LABELLED_TOKEN =
  /\b(?:api[\s_-]?key|key|token|secret|password|bearer)\b\s*(?:is\s+|[:=]\s*)?["'`]?([A-Za-z0-9._~+/=-]{16,})/gi

export interface KeySpan {
  start: number
  end: number
}

function looksLikeSecret(token: string): boolean {
  return /[0-9]/.test(token) && /[A-Za-z]/.test(token)
}

export function findApiKeys(text: string): KeySpan[] {
  const spans: KeySpan[] = []
  for (const pattern of KEY_PATTERNS) {
    for (const match of text.matchAll(pattern)) {
      spans.push({ start: match.index, end: match.index + match[0].length })
    }
  }
  for (const match of text.matchAll(LABELLED_TOKEN)) {
    const token = match[1]
    if (!token || !looksLikeSecret(token)) continue
    const end = match.index + match[0].length
    spans.push({ start: end - token.length, end })
  }
  spans.sort((a, b) => a.start - b.start)
  const merged: KeySpan[] = []
  for (const span of spans) {
    const last = merged[merged.length - 1]
    if (last && span.start <= last.end) last.end = Math.max(last.end, span.end)
    else merged.push({ ...span })
  }
  return merged
}

export const REMOVED_KEY_PLACEHOLDER = "[key removed]"

export function redactApiKeys(text: string): string {
  let out = ""
  let cursor = 0
  for (const span of findApiKeys(text)) {
    out += text.slice(cursor, span.start) + REMOVED_KEY_PLACEHOLDER
    cursor = span.end
  }
  return out + text.slice(cursor)
}
