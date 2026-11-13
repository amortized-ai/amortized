import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { CustomEndpointsCard } from "./custom-endpoints-card"

function renderCard() {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <CustomEndpointsCard />
    </QueryClientProvider>,
  )
}

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })
}

describe("CustomEndpointsCard", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("shows a load failure with a retry instead of hiding the card", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(json(500, { error: "boom" }))
      .mockResolvedValueOnce(json(200, { endpoints: [{ name: "my-vllm", baseURL: "https://v.example.com/v1" }] }))
    vi.stubGlobal("fetch", fetchMock)
    renderCard()

    expect(await screen.findByText(/Could not load custom endpoints/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /Retry/ }))
    expect(await screen.findByText("my-vllm")).toBeInTheDocument()
  })

  it("asks before removing an endpoint", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(json(200, { endpoints: [{ name: "my-vllm", baseURL: "https://v.example.com/v1" }] }))
    vi.stubGlobal("fetch", fetchMock)
    renderCard()

    fireEvent.click(await screen.findByRole("button", { name: "Remove my-vllm" }))
    expect(screen.getByText(/removes "my-vllm" and its stored API key/)).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(false)

    fireEvent.click(screen.getByRole("button", { name: /Delete/ }))
    await waitFor(() =>
      expect(fetchMock.mock.calls.some(([, init]) => init?.method === "DELETE")).toBe(true),
    )
  })
})
