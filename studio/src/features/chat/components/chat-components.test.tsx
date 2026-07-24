import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { OptionCards } from "./option-cards"
import { ToolBadge } from "./tool-badge"
import { ActionCard } from "./action-card"
import { PlanProgress } from "./plan-progress"
import { ChatInput } from "./chat-input"
import { MessageList } from "./message-list"
import { MessageBubble } from "./message-bubble"
import type { ChatMessage, PhasePlan } from "../types"

describe("OptionCards", () => {
  const cards = [
    { title: "Customer support", description: "Classify tickets", value: "support" },
    { title: "IT helpdesk", description: "Route requests", value: "it" },
  ]

  it("renders all cards", () => {
    render(<OptionCards cards={cards} onSelect={vi.fn()} />)
    expect(screen.getByText("Customer support")).toBeInTheDocument()
    expect(screen.getByText("IT helpdesk")).toBeInTheDocument()
  })

  it("calls onSelect with correct value when clicked", () => {
    const onSelect = vi.fn()
    render(<OptionCards cards={cards} onSelect={onSelect} />)

    fireEvent.click(screen.getByText("Customer support"))
    expect(onSelect).toHaveBeenCalledWith("support")
  })

  it("calls onSelect on Enter key", () => {
    const onSelect = vi.fn()
    render(<OptionCards cards={cards} onSelect={onSelect} />)

    const card = screen.getByText("IT helpdesk").closest("button")!
    fireEvent.keyDown(card, { key: "Enter" })
    expect(onSelect).toHaveBeenCalledWith("it")
  })
})

describe("ToolBadge", () => {
  const toolResult = {
    name: "search_docs",
    result: "Found 3 documents",
    collapsed: true,
  }

  it("renders tool name in badge", () => {
    render(<ToolBadge toolResult={toolResult} />)
    expect(screen.getByText("search docs")).toBeInTheDocument()
  })

  it("starts collapsed and does not show result", () => {
    render(<ToolBadge toolResult={toolResult} />)
    expect(screen.queryByText("Found 3 documents")).not.toBeInTheDocument()
  })

  it("expands on click to show result", () => {
    render(<ToolBadge toolResult={toolResult} />)
    fireEvent.click(screen.getByText("search docs"))
    fireEvent.click(screen.getByText("search_docs"))
    expect(screen.getByText("Found 3 documents")).toBeInTheDocument()
  })

  it("collapses again on second click", () => {
    render(<ToolBadge toolResult={toolResult} />)
    fireEvent.click(screen.getByText("search docs"))
    const detailButton = screen.getByText("search_docs")
    fireEvent.click(detailButton)
    expect(screen.getByText("Found 3 documents")).toBeInTheDocument()

    fireEvent.click(detailButton)
    expect(screen.queryByText("Found 3 documents")).not.toBeInTheDocument()
  })
})

describe("ActionCard", () => {
  const action = {
    action: "Submit training job",
    description: "Train llama-3 on customer_tickets dataset",
    params: {},
  }

  it("renders action title and description", () => {
    render(
      <ActionCard action={action} onConfirm={vi.fn()} onReject={vi.fn()} />,
    )
    expect(screen.getByText("Submit training job")).toBeInTheDocument()
    expect(screen.getByText(/Train llama-3/)).toBeInTheDocument()
  })

  it("calls onConfirm when confirm button is clicked", () => {
    const onConfirm = vi.fn()
    render(
      <ActionCard action={action} onConfirm={onConfirm} onReject={vi.fn()} />,
    )
    fireEvent.click(screen.getByText("Confirm"))
    expect(onConfirm).toHaveBeenCalledOnce()
  })

  it("calls onReject when reject button is clicked", () => {
    const onReject = vi.fn()
    render(
      <ActionCard action={action} onConfirm={vi.fn()} onReject={onReject} />,
    )
    fireEvent.click(screen.getByText("Reject"))
    expect(onReject).toHaveBeenCalledOnce()
  })
})

describe("PlanProgress", () => {
  const plan: PhasePlan = {
    phase: "sdg",
    label: "Data Generation",
    steps: [
      { label: "Gather requirements", status: "completed" },
      { label: "Load recipe", status: "active" },
      { label: "Estimate cost", status: "pending" },
      { label: "Generate data", status: "pending" },
    ],
  }

  it("renders nothing when no plan", () => {
    const { container } = render(<PlanProgress plan={null} />)
    expect(container.firstChild).toBeNull()
  })

  it("shows phase label and active step in header", () => {
    render(<PlanProgress plan={plan} />)
    expect(screen.getByText("Data Generation")).toBeInTheDocument()
    expect(screen.getAllByText("Load recipe").length).toBeGreaterThanOrEqual(1)
  })

  it("renders correct number of progress dots", () => {
    const { container } = render(<PlanProgress plan={plan} />)
    const dots = container.querySelectorAll(".h-2.w-2.rounded-full")
    expect(dots).toHaveLength(4)
  })
})

describe("ChatInput", () => {
  it("calls onSend with trimmed value on Enter", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />)

    const textarea = screen.getByPlaceholderText("Type a message...")
    fireEvent.change(textarea, { target: { value: "  hello  " } })
    fireEvent.keyDown(textarea, { key: "Enter" })

    expect(onSend).toHaveBeenCalledWith("hello")
  })

  it("does not send on Shift+Enter", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />)

    const textarea = screen.getByPlaceholderText("Type a message...")
    fireEvent.change(textarea, { target: { value: "hello" } })
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true })

    expect(onSend).not.toHaveBeenCalled()
  })

  it("is disabled when disabled prop is true", () => {
    render(<ChatInput onSend={vi.fn()} disabled />)
    const textarea = screen.getByPlaceholderText("Type a message...")
    expect(textarea).toBeDisabled()
  })

  it("does not send empty messages", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />)

    const textarea = screen.getByPlaceholderText("Type a message...")
    fireEvent.change(textarea, { target: { value: "   " } })
    fireEvent.keyDown(textarea, { key: "Enter" })

    expect(onSend).not.toHaveBeenCalled()
  })

  it("clears input after sending", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />)

    const textarea = screen.getByPlaceholderText("Type a message...") as HTMLTextAreaElement
    fireEvent.change(textarea, { target: { value: "hello" } })
    fireEvent.keyDown(textarea, { key: "Enter" })

    expect(textarea.value).toBe("")
  })
})

describe("MessageList", () => {
  it("shows welcome screen when no messages", () => {
    render(<MessageList messages={[]} />)
    expect(
      screen.getByText("Welcome to Amortized Studio"),
    ).toBeInTheDocument()
  })

  it("renders user and assistant messages", () => {
    const messages: ChatMessage[] = [
      {
        id: "1",
        role: "user",
        content: "Hello",
        timestamp: new Date().toISOString(),
        toolResults: [],
        proposedAction: null,
        optionCards: [],
      },
      {
        id: "2",
        role: "assistant",
        content: "Hi there!",
        timestamp: new Date().toISOString(),
        toolResults: [],
        proposedAction: null,
        optionCards: [],
      },
    ]

    render(<MessageList messages={messages} />)
    expect(screen.getByText("Hello")).toBeInTheDocument()
    expect(screen.getByText("Hi there!")).toBeInTheDocument()
  })

  it("auto-scrolls to bottom on new messages", () => {
    vi.useFakeTimers()
    const scrollIntoViewMock = vi.fn()
    window.HTMLElement.prototype.scrollIntoView = scrollIntoViewMock

    const messages: ChatMessage[] = [
      {
        id: "1",
        role: "user",
        content: "test",
        timestamp: new Date().toISOString(),
        toolResults: [],
        proposedAction: null,
        optionCards: [],
      },
    ]

    render(<MessageList messages={messages} />)
    vi.advanceTimersByTime(300)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ behavior: "smooth", block: "end" })
    vi.useRealTimers()
  })
})

describe("MessageBubble — parsed option cards", () => {
  function makeMsg(content: string): ChatMessage {
    return {
      id: "1",
      role: "assistant",
      content,
      timestamp: new Date().toISOString(),
      toolResults: [],
      proposedAction: null,
      optionCards: [],
    }
  }

  it("renders clickable option cards from numbered options", () => {
    const onSelect = vi.fn()
    render(
      <MessageBubble
        message={makeMsg("Pick one:\n1) 80/20 split\n2) 70/30 split\n3) 90/10 split")}
        onOptionSelect={onSelect}
      />,
    )
    const cards = screen.getAllByRole("button")
    expect(cards).toHaveLength(4) // 3 options + "Other"
    expect(cards[0]).toHaveTextContent("80/20 split")
    expect(cards[1]).toHaveTextContent("70/30 split")
    expect(cards[2]).toHaveTextContent("90/10 split")

    fireEvent.click(cards[0]!)
    expect(onSelect).toHaveBeenCalledWith("80/20 split")
  })

  it("renders clickable option cards from bullet options", () => {
    const onSelect = vi.fn()
    render(
      <MessageBubble
        message={makeMsg("Choose:\n- Option A\n- Option B")}
        onOptionSelect={onSelect}
      />,
    )
    const cards = screen.getAllByRole("button")
    expect(cards).toHaveLength(3) // 2 options + "Other"
    expect(cards[0]).toHaveTextContent("Option A")
    expect(cards[1]).toHaveTextContent("Option B")
  })

  it("does not parse options from user messages", () => {
    const userMsg: ChatMessage = {
      ...makeMsg("1) First\n2) Second"),
      role: "user",
    }
    render(<MessageBubble message={userMsg} onOptionSelect={vi.fn()} />)
    expect(screen.queryAllByRole("button")).toHaveLength(0)
  })

  it("renders a single option as a clickable card", () => {
    render(
      <MessageBubble
        message={makeMsg("Just one:\n1) Only option — the sole choice")}
        onOptionSelect={vi.fn()}
      />,
    )
    expect(screen.queryAllByRole("button").length).toBeGreaterThanOrEqual(1)
  })
})
