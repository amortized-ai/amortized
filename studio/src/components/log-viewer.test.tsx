import { render, screen } from "@testing-library/react"
import { describe, it, expect } from "vitest"
import { LogViewer } from "./log-viewer"

describe("LogViewer", () => {
  it("renders log lines", () => {
    render(<LogViewer lines={["line 1", "line 2", "line 3"]} />)
    expect(screen.getByText("line 1")).toBeInTheDocument()
    expect(screen.getByText("line 2")).toBeInTheDocument()
    expect(screen.getByText("line 3")).toBeInTheDocument()
  })

  it("renders empty when no lines", () => {
    const { container } = render(<LogViewer lines={[]} />)
    const viewer = container.querySelector("[data-testid='log-viewer']")
    expect(viewer).toBeInTheDocument()
    expect(viewer?.children.length).toBe(0)
  })

  it("shows streaming cursor when isStreaming", () => {
    const { container } = render(<LogViewer lines={["test"]} isStreaming />)
    const cursor = container.querySelector(".animate-pulse")
    expect(cursor).toBeInTheDocument()
  })

  it("hides streaming cursor when not streaming", () => {
    const { container } = render(<LogViewer lines={["test"]} isStreaming={false} />)
    const cursor = container.querySelector(".animate-pulse")
    expect(cursor).not.toBeInTheDocument()
  })
})
