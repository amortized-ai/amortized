import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

vi.mock("@/stores/settings-store", () => ({
  useSettingsStore: {
    getState: () => ({ apiKey: "test-key-123" }),
  },
}))

vi.mock("@/stores/chat-store", () => ({
  useChatStore: {
    getState: () => ({
      getSessionId: vi.fn(() => undefined),
      setSessionId: vi.fn(),
      clearSessionId: vi.fn(),
      setSessionStatus: vi.fn(),
      getSessionStatus: vi.fn(() => "unknown"),
      getConversationMessages: vi.fn(() => []),
    }),
  },
}))

import {
  getJobs,
  getJob,
  createJob,
  cancelJob,
  getHealth,
  getConfig,
  getJobLogs,
  searchMlflowRuns,
  getMlflowGatewayRoutes,
  ApiError,
} from "@/lib/api-client"

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

function errorResponse(status: number, body: unknown) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      statusText: "Bad Request",
      headers: { "Content-Type": "application/json" },
    }),
  )
}

describe("API Client", () => {
  describe("request formatting", () => {
    it("includes auth header when apiKey is set", async () => {
      mockFetch.mockReturnValue(jsonResponse([]))
      await getJobs()

      const [, init] = mockFetch.mock.calls[0]!
      expect(init.headers).toHaveProperty("Authorization", "Bearer test-key-123")
    })

    it("sets Content-Type to application/json", async () => {
      mockFetch.mockReturnValue(jsonResponse([]))
      await getJobs()

      const [, init] = mockFetch.mock.calls[0]!
      expect(init.headers).toHaveProperty("Content-Type", "application/json")
    })

    it("sends GET requests to the correct URL", async () => {
      mockFetch.mockReturnValue(jsonResponse([]))
      await getJobs()

      const [url] = mockFetch.mock.calls[0]!
      expect(url).toBe("/api/v1/jobs")
    })

    it("appends query params for filters", async () => {
      mockFetch.mockReturnValue(jsonResponse([]))
      await getJobs({ type: "training", status: "running" })

      const [url] = mockFetch.mock.calls[0]!
      expect(url).toContain("type=training")
      expect(url).toContain("status=running")
    })
  })

  describe("GET endpoints", () => {
    it("getJob fetches a single job", async () => {
      const job = { id: "j1", type: "training", status: "running" }
      mockFetch.mockReturnValue(jsonResponse(job))

      const result = await getJob("j1")
      expect(result).toEqual(job)
      expect(mockFetch.mock.calls[0]![0]).toBe("/api/v1/jobs/j1")
    })

    it("getHealth returns health data", async () => {
      const health = { status: "ok", timestamp: "2026-06-10T12:00:00Z" }
      mockFetch.mockReturnValue(jsonResponse(health))

      const result = await getHealth()
      expect(result).toEqual(health)
    })

    it("getConfig returns config data", async () => {
      const config = { mlflow_tracking_uri: "http://mlflow:5000", version: "0.3.0" }
      mockFetch.mockReturnValue(jsonResponse(config))

      const result = await getConfig()
      expect(result).toEqual(config)
    })

    it("getJobLogs returns log lines", async () => {
      const logs = ["line 1", "line 2", "line 3"]
      mockFetch.mockReturnValue(jsonResponse({ job_id: "j1", logs, message: "ok" }))

      const result = await getJobLogs("j1")
      expect(result).toEqual(logs)
      expect(mockFetch.mock.calls[0]![0]).toBe("/api/v1/jobs/j1/logs?tail=2000")
    })
  })

  describe("POST endpoints", () => {
    it("createJob sends POST with body", async () => {
      const config = { type: "training", config: { model: "llama" } }
      const created = { id: "j2", ...config, status: "queued" }
      mockFetch.mockReturnValue(jsonResponse(created))

      const result = await createJob(config)
      expect(result).toEqual(created)

      const [url, init] = mockFetch.mock.calls[0]!
      expect(url).toBe("/api/v1/jobs")
      expect(init.method).toBe("POST")
      expect(JSON.parse(init.body as string)).toEqual(config)
    })

    it("cancelJob sends DELETE", async () => {
      const cancelled = { id: "j1", status: "cancelled" }
      mockFetch.mockReturnValue(jsonResponse(cancelled))

      await cancelJob("j1")

      const [url, init] = mockFetch.mock.calls[0]!
      expect(url).toBe("/api/v1/jobs/j1")
      expect(init.method).toBe("DELETE")
    })

    it("searchMlflowRuns sends POST to MLflow API", async () => {
      const body = { experiment_ids: ["1"], max_results: 10 }
      const response = { runs: [] }
      mockFetch.mockReturnValue(jsonResponse(response))

      const result = await searchMlflowRuns(body)
      expect(result).toEqual(response)

      const [url, init] = mockFetch.mock.calls[0]!
      expect(url).toBe("/mlflow/api/2.0/mlflow/runs/search")
      expect(init.method).toBe("POST")
      expect(JSON.parse(init.body as string)).toEqual(body)
    })
  })

  describe("MLflow gateway endpoints", () => {
    it("getMlflowGatewayRoutes fetches routes", async () => {
      const endpointsResponse = {
        endpoints: [{
          name: "gpt-4o",
          endpoint_id: "ep-1",
          model_mappings: [{
            linkage_type: "PRIMARY",
            model_definition: { model_name: "gpt-4o", provider: "openai" },
          }],
        }],
      }
      mockFetch.mockReturnValue(jsonResponse(endpointsResponse))

      const result = await getMlflowGatewayRoutes()
      expect(result).toEqual({
        routes: [{
          name: "gpt-4o",
          route_type: "llm/v1/chat",
          model: { name: "gpt-4o", provider: "openai" },
          endpoint_id: "ep-1",
        }],
      })
      expect(mockFetch.mock.calls[0]![0]).toBe("/mlflow/api/3.0/mlflow/gateway/endpoints/list")
    })
  })

  describe("error handling", () => {
    it("throws ApiError on non-2xx response", async () => {
      mockFetch.mockReturnValue(errorResponse(400, { detail: "bad request" }))
      await expect(getJobs()).rejects.toThrow(ApiError)
    })

    it("includes status and body in ApiError", async () => {
      mockFetch.mockReturnValue(
        errorResponse(404, { detail: "not found" }),
      )

      try {
        await getJob("nonexistent")
        expect.fail("should have thrown")
      } catch (err) {
        expect(err).toBeInstanceOf(ApiError)
        const apiErr = err as InstanceType<typeof ApiError>
        expect(apiErr.status).toBe(404)
        expect(apiErr.body).toEqual({ detail: "not found" })
      }
    })
  })
})
