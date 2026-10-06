import { describe, it, expect } from "vitest"

import { summarizeConversation } from "@/lib/context-summarizer"
import type { PersistedMessage } from "@/stores/chat-store"

function msg(partial: Partial<PersistedMessage>): PersistedMessage {
  return { id: "m", role: "assistant", content: "", timestamp: "t", ...partial }
}

describe("summarizeConversation", () => {
  it("returns empty string when there are no messages", () => {
    expect(summarizeConversation([])).toBe("")
  })

  it("replays the conversation as [User]/[Assistant] lines", () => {
    const out = summarizeConversation([
      msg({ role: "user", content: "train a model" }),
      msg({ role: "assistant", content: "Sure — here is how." }),
    ])
    expect(out).toContain("[User] train a model")
    expect(out).toContain("[Assistant] Sure — here is how.")
  })

  it("marks a failed (error-flagged) turn without replaying the error text", () => {
    const out = summarizeConversation([
      msg({ role: "user", content: "train a model" }),
      msg({ role: "assistant", content: "Sure — here is how." }),
      msg({ role: "user", content: "use claude via vertex" }),
      msg({
        role: "assistant",
        content: "Error: That model isn't available with your API key or project (auth/permission).",
        error: true,
      }),
    ])
    // the user's turns and the real assistant reply survive...
    expect(out).toContain("[User] train a model")
    expect(out).toContain("[Assistant] Sure — here is how.")
    expect(out).toContain("[User] use claude via vertex")
    // ...the failed turn is kept as a neutral marker (history stays accurate)...
    expect(out).toContain("(previous attempt failed to produce a response)")
    // ...but the raw error text is never replayed as assistant content
    expect(out).not.toContain("Error:")
    expect(out).not.toContain("isn't available")
  })
})
