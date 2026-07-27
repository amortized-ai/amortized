import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { MemoryRouter } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { TooltipProvider } from "@/components/ui/tooltip"
import { AppLayout } from "@/app/layout"

vi.mock("@/features/settings/api/use-settings", () => ({
  useHealth: () => ({ data: { status: "ok" }, isError: false, isLoading: false }),
}))

vi.mock("@/features/jobs/api/use-jobs", () => ({
  useJobs: () => ({ data: [], isLoading: false }),
}))

function renderLayout(initialRoute = "/chat") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <MemoryRouter initialEntries={[initialRoute]}>
          <AppLayout />
        </MemoryRouter>
      </TooltipProvider>
    </QueryClientProvider>,
  )
}

describe("AppLayout", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the sidebar with nav items", () => {
    renderLayout()
    const navLabels = ["Overview", "Chat", "Jobs", "Datasets", "Models", "Recipes", "Settings"]
    for (const label of navLabels) {
      expect(screen.getAllByText(label).length).toBeGreaterThanOrEqual(1)
    }
  })

  it("renders breadcrumb navigation in the top bar", () => {
    renderLayout()
    expect(screen.getByRole("navigation", { name: /breadcrumb/i })).toBeInTheDocument()
  })

  it("renders the connection status indicator", () => {
    renderLayout()
    const dot = screen.getByTestId("connection-dot")
    expect(dot).toBeInTheDocument()
    expect(dot.className).toContain("bg-rh-green")
  })

  it("renders the sidebar toggle button", () => {
    renderLayout()
    expect(screen.getByTestId("sidebar-trigger")).toBeInTheDocument()
    expect(screen.getByText("Toggle Sidebar")).toBeInTheDocument()
  })

  it("sidebar trigger has higher z-index than sidebar for clickability", () => {
    renderLayout()
    const trigger = screen.getByTestId("sidebar-trigger")
    const header = trigger.closest("header")
    expect(header?.className).toContain("z-20")
  })

  it("highlights active nav item based on route", () => {
    renderLayout("/jobs")
    const sidebarEl = document.querySelector("[data-sidebar='sidebar']")!
    const jobsLink = Array.from(sidebarEl.querySelectorAll("a")).find(a => a.textContent?.trim() === "Jobs")
    const chatLink = Array.from(sidebarEl.querySelectorAll("a")).find(a => a.textContent?.trim() === "Chat")
    expect(jobsLink?.closest("[data-active]")?.getAttribute("data-active")).toBe("true")
    expect(chatLink?.closest("[data-active]")?.getAttribute("data-active")).toBe("false")
  })

  it("renders the theme toggle button", () => {
    renderLayout()
    expect(screen.getByTestId("theme-toggle")).toBeInTheDocument()
  })

  it("toggles dark class on document when theme toggle is clicked", () => {
    renderLayout()
    const toggle = screen.getByTestId("theme-toggle")
    expect(document.documentElement.classList.contains("dark")).toBe(false)
    fireEvent.click(toggle)
    expect(document.documentElement.classList.contains("dark")).toBe(true)
    fireEvent.click(toggle)
    expect(document.documentElement.classList.contains("dark")).toBe(false)
  })
})
