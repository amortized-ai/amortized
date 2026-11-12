import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

// A stateful session map, so the tests can see whether a send keeps or abandons its session.
const sessions = new Map<string, string>()
const cleared: string[] = []

vi.mock("@/stores/settings-store", () => ({
  useSettingsStore: { getState: () => ({ apiKey: "" }) },
}))

vi.mock("@/stores/chat-store", () => ({
  useChatStore: {
    getState: () => ({
      getSessionId: (conv: string) => sessions.get(conv),
      setSessionId: (conv: string, id: string) => { sessions.set(conv, id) },
      clearSessionId: (conv: string) => { cleared.push(conv); sessions.delete(conv) },
      setSessionStatus: () => {},
      getSessionStatus: () => "unknown",
      getConversationMessages: () => [],
    }),
  },
}))

import { sendOpenCodeMessage } from "@/lib/api-client"

const EVENT = { eventId: "confirmation_blocked:m1" }
const mockFetch = vi.fn()
const originalFetch = globalThis.fetch
let created = 0
const messagePosts: Array<{ session: string; eventId: unknown }> = []

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status })
}

// Routes session creation and turn polling; `onMessage` decides each message POST's fate.
function route(onMessage: (n: number, session: string) => Response | Promise<Response>, onTurn?: (turn: string) => Response) {
  mockFetch.mockImplementation(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/agent/session") && init?.method === "POST") {
      created++
      return json({ id: `ses-${created}` })
    }
    const message = url.match(/\/agent\/session\/([^/]+)\/message$/)
    if (message) {
      const session = message[1]!
      messagePosts.push({ session, eventId: JSON.parse(String(init?.body)).event_id })
      return onMessage(messagePosts.length, session)
    }
    const turn = url.match(/\/turn\/([^/]+)$/)
    if (turn) return onTurn ? onTurn(turn[1]!) : json({ active: false, result: { info: {}, parts: [] } })
    throw new Error(`unexpected ${url}`)
  })
}

function send(options = EVENT) {
  return sendOpenCodeMessage("conv", "blocked", undefined, "confirmation_blocked", undefined, "eval", undefined, options)
}

beforeEach(() => {
  globalThis.fetch = mockFetch
  mockFetch.mockReset()
  sessions.clear()
  cleared.length = 0
  messagePosts.length = 0
  created = 0
})

afterEach(() => {
  globalThis.fetch = originalFetch
})

describe("sendOpenCodeMessage — idempotent events keep their session", () => {
  it("a manual retry after exhausted lost responses goes to the original session", async () => {
    // Every response is lost (the server may have started the turn each time).
    route(() => { throw new TypeError("fetch failed") })
    await expect(send()).rejects.toThrow()
    expect(cleared).toEqual([])

    // The user clicks Retry: it must reach the same session, which dedupes the event.
    route(() => json({ turn_id: "t1" }))
    await send()

    expect(created).toBe(1)
    expect(new Set(messagePosts.map((p) => p.session))).toEqual(new Set(["ses-1"]))
    expect(new Set(messagePosts.map((p) => p.eventId))).toEqual(new Set(["confirmation_blocked:m1"]))
  })

  it("retries a failed turn on the same session", async () => {
    route((n) => json({ turn_id: `t${n}` }), (turn) =>
      turn === "t1"
        ? json({ active: false, error: "provider error", error_status: 500 })
        : json({ active: false, result: { info: {}, parts: [] } }),
    )
    await send()
    expect(created).toBe(1)
    expect(messagePosts.map((p) => p.session)).toEqual(["ses-1", "ses-1"])
  })

  it("keeps the session even when a failed turn exhausts the retries", async () => {
    route((n) => json({ turn_id: `t${n}` }), () => json({ active: false, error: "boom", error_status: 500 }))
    await expect(send()).rejects.toThrow()
    expect(cleared).toEqual([])
    expect(sessions.get("conv")).toBe("ses-1")
  })

  it("moves to a new session only when the server no longer knows it (404)", async () => {
    route((_n, session) => (session === "ses-1" ? json({}, 404) : json({ turn_id: "t1" })))
    await send()
    expect(messagePosts.map((p) => p.session)).toEqual(["ses-1", "ses-2"])
  })
})

describe("sendOpenCodeMessage — ordinary sends are unchanged", () => {
  it("still resets the session after a network error", async () => {
    route((n) => {
      if (n === 1) throw new TypeError("fetch failed")
      return json({ turn_id: "t1" })
    })
    await send({ eventId: undefined } as never)
    expect(created).toBe(2)
    expect(messagePosts.map((p) => p.session)).toEqual(["ses-1", "ses-2"])
  })
})
