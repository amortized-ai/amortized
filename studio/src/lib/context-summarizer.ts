import type { PersistedMessage } from "@/stores/chat-store"

const SOFT_CAP = 100_000
const MAX_TOOL_RESULT_CHARS = 2000

export function summarizeConversation(messages: PersistedMessage[]): string {
  if (messages.length === 0) return ""

  const lines: string[] = []

  for (const msg of messages) {
    // A failed turn's content is the raw error text. Don't replay that verbatim (it would read as
    // the assistant having "said" an error), but keep the history accurate with a neutral marker
    // rather than dropping the turn silently.
    if (msg.error) {
      lines.push("[Assistant] (previous attempt failed to produce a response)")
      lines.push("")
      continue
    }
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

  // Framed as a READ-ONLY record, not a transcript to continue. Telling the model to
  // "continue the conversation" from a role-labeled history induces it to continue the
  // *transcript* — fabricating the next "[User]" turn and self-answering, which then renders
  // as an inline "user <option value>" leak inside an assistant bubble (#556). The history is
  // context, so it is framed as context; the model responds only to the user's next real message.
  return [
    "[System note: you are RESUMING an earlier conversation whose live session was",
    "lost. The block below is a READ-ONLY record of what was already said, provided",
    "only so you keep full context (the user's task, jobs submitted, choices made,",
    "workflow progress). It is NOT a live transcript to continue: do NOT reproduce",
    "it, do NOT quote it back, and do NOT write any further [User]/[Assistant] turns.",
    "Read it, then wait for and respond only to the user's next message.]",
    "",
    "--- Conversation record (read-only) ---",
    "",
    body,
    "--- End of record ---",
  ].join("\n")
}
