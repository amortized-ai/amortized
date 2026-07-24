import type { PersistedMessage } from "@/stores/chat-store"

const MAX_CHARS = 6000

export function summarizeConversation(messages: PersistedMessage[]): string {
  if (messages.length === 0) return ""

  const lines: string[] = []
  lines.push("[Context from previous conversation — the session was lost. Here is a summary of what was discussed:]")
  lines.push("")

  let charCount = 0
  const pairs: string[] = []

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!
    let line: string

    if (msg.role === "user") {
      line = `User: ${msg.content.slice(0, 200)}`
    } else {
      let content = msg.content.slice(0, 300)
      if (msg.content.length > 300) content += "..."
      line = `Assistant: ${content}`

      if (msg.toolResults?.length) {
        const toolSummary = msg.toolResults
          .map(t => `[Tool: ${t.name}]`)
          .join(", ")
        line += `\n  ${toolSummary}`
      }
    }

    if (charCount + line.length > MAX_CHARS) break
    charCount += line.length
    pairs.unshift(line)
  }

  lines.push(...pairs)
  lines.push("")
  lines.push("[End of context. Continue the conversation naturally from where it left off.]")

  return lines.join("\n")
}
