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
  it("returns ok when health and mlflow endpoints succeed", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health"))
        return jsonResponse({ status: "ok", timestamp: "2026-06-10T12:00:00Z" })
      if (url.includes("/mlflow/"))
        return jsonResponse({ runs: [] })
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("ok")
      expect(result.current.mlflow).toBe("ok")
    })
  })

  it("returns error when health endpoint fails", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health")) return errorResponse()
      if (url.includes("/mlflow/")) return jsonResponse({ runs: [] })
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("error")
      expect(result.current.mlflow).toBe("ok")
    })
  })

  it("returns error when mlflow endpoint fails", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health"))
        return jsonResponse({ status: "ok", timestamp: "2026-06-10T12:00:00Z" })
      if (url.includes("/mlflow/")) return errorResponse()
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("ok")
      expect(result.current.mlflow).toBe("error")
    })
  })

  it("returns loading initially", () => {
    mockFetch.mockImplementation(() => new Promise(() => {}))

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    expect(result.current.backend).toBe("loading")
    expect(result.current.mlflow).toBe("loading")
  })

  it("only returns backend and mlflow status", async () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes("/health"))
        return jsonResponse({ status: "ok", timestamp: "2026-06-10T12:00:00Z" })
      if (url.includes("/mlflow/"))
        return jsonResponse({ runs: [] })
      return jsonResponse({})
    })

    const { result } = renderHook(() => useSystemHealth(), { wrapper: createWrapper() })

    await waitFor(() => {
      expect(result.current.backend).toBe("ok")
    })

    expect(Object.keys(result.current).sort()).toEqual(["backend", "mlflow"])
  })
})
