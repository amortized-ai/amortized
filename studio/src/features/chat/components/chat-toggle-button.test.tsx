import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { ChatToggleButton } from "./chat-toggle-button"

const mockTogglePanel = vi.fn()
vi.mock("@/stores/chat-store", () => ({
  useChatStore: () => ({
    togglePanel: mockTogglePanel,
    panelOpen: false,
    setPanelOpen: vi.fn(),
  }),
}))

vi.mock("./chat-panel", () => ({
  ChatPanel: () => <div data-testid="chat-panel" />,
}))

describe("ChatToggleButton", () => {
  it("renders the toggle button", () => {
    render(<ChatToggleButton />)
    expect(screen.getByTestId("chat-toggle-button")).toBeInTheDocument()
  })

  it("calls togglePanel when clicked", () => {
    render(<ChatToggleButton />)
    fireEvent.click(screen.getByTestId("chat-toggle-button"))
    expect(mockTogglePanel).toHaveBeenCalledOnce()
  })

  it("renders the ChatPanel", () => {
    render(<ChatToggleButton />)
    expect(screen.getByTestId("chat-panel")).toBeInTheDocument()
  })

  it("has correct aria-label", () => {
    render(<ChatToggleButton />)
    expect(screen.getByLabelText("Toggle chat")).toBeInTheDocument()
  })
})
