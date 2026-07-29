import type { PersistedMessage } from "@/stores/chat-store"

const SOFT_CAP = 100_000
const MAX_TOOL_RESULT_CHARS = 2000

export function summarizeConversation(messages: PersistedMessage[]): string {
  if (messages.length === 0) return ""

  const lines: string[] = []

  for (const msg of messages) {
    if (msg.role === "user") {
      lines.push(`[User] ${msg.content}`)
    } else {
      lines.push(`[Assistant] ${msg.content}`)
      if (msg.toolResults?.length) {
        for (const t of msg.toolResults) {
          const result = t.result
            ? t.result.length > MAX_TOOL_RESULT_CHARS
              ? t.result.slice(0, MAX_TOOL_RESULT_CHARS) + "..."
              : t.result
            : ""
          lines.push(`  [Tool: ${t.name} → ${result}]`)
        }
      }
    }
    lines.push("")
  }

  let body = lines.join("\n")

  if (body.length > SOFT_CAP) {
    const trimmed: string[] = []
    let charCount = 0
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i]!
      if (charCount + line.length > SOFT_CAP) break
      charCount += line.length
      trimmed.unshift(line)
    }
    body = trimmed.join("\n")
  }

  return [
    "[System: This is a restored conversation session. Below is the complete",
    "message history from before the session was lost. Continue naturally as",
    "if this is your ongoing conversation — maintain all context about the",
    "user's task, jobs submitted, choices made, and workflow progress.]",
    "",
    "--- Message History ---",
    "",
    body,
    "--- End of History ---",
    "",
    "Continue the conversation from here.",
  ].join("\n")
}
