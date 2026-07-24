import { renderHook, waitFor } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { createElement, type ReactNode } from "react"
import { useSystemHealth } from "@/hooks/use-system-health"

vi.mock("@/stores/settings-store", () => ({
  useSettingsStore: { getState: () => ({ apiKey: "test" }) },
}))

const mockFetch = vi.fn()
const originalFetch = globalThis.fetch

beforeEach(() => {
  globalThis.fetch = mockFetch
  mockFetch.mockReset()
})

afterEach(() => {
  globalThis.fetch = originalFetch
})

function jsonResponse(data: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(data), {
      status,
      statusText: status === 200 ? "OK" : "Error",
      headers: { "Content-Type": "application/json" },
    }),
  )
}

function errorResponse() {
  return Promise.resolve(
    new Response("Internal Server Error", { status: 500, statusText: "Error" }),
  )
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client: queryClient }, children)
  }
}

describe("useSystemHealth", () => {
  it("returns ok when health endpoint succeeds", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health"))
        return jsonResponse({ status: "ok", timestamp: "2026-06-10T12:00:00Z" })
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("ok")
    })
  })

  it("returns error when health endpoint fails", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health")) return errorResponse()
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("error")
    })
  })

  it("returns loading initially", () => {
    mockFetch.mockImplementation(() => new Promise(() => {}))

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    expect(result.current.backend).toBe("loading")
  })

  it("only returns backend status (no apiKeys, computeBackends, gpu)", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health"))
        return jsonResponse({ status: "ok", timestamp: "2026-06-10T12:00:00Z" })
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("ok")
    })

    expect(Object.keys(result.current)).toEqual(["backend"])
  })
})
