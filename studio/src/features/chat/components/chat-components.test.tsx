import { render, screen, fireEvent } from "@testing-library/react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"

const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
function Wrapper({ children }: { children: React.ReactNode }) {
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
}
import { OptionCards } from "./option-cards"
import { ToolBadge } from "./tool-badge"
import { ActionCard } from "./action-card"
import { PlanProgress } from "./plan-progress"
import { ChatInput } from "./chat-input"
import { MessageList } from "./message-list"
import { MessageBubble } from "./message-bubble"
import type { ChatMessage, PhasePlan } from "../types"
import { useChatStore } from "@/stores/chat-store"

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
    fireEvent.click(screen.getByText("Cancel"))
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
  beforeEach(() => {
    useChatStore.setState({ currentConversationId: "test-conv", conversations: [], drafts: {} })
  })

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
    render(<MessageList messages={[]} />, { wrapper: Wrapper })
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

    render(<MessageList messages={messages} />, { wrapper: Wrapper })
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

    render(<MessageList messages={messages} />, { wrapper: Wrapper })
    vi.advanceTimersByTime(300)
    expect(scrollIntoViewMock).toHaveBeenCalledWith({ behavior: "smooth", block: "end" })
    vi.useRealTimers()
  })
})

describe("MessageBubble — structured option cards", () => {
  function makeMsg(content: string, toolResults: ChatMessage["toolResults"] = []): ChatMessage {
    return {
      id: "1",
      role: "assistant",
      content,
      timestamp: new Date().toISOString(),
      toolResults,
      proposedAction: null,
      optionCards: [],
    }
  }

  it("renders option cards from present_options tool result", () => {
    const onSelect = vi.fn()
    const toolResults = [{
      name: "present_options",
      result: JSON.stringify({
        options: [
          { title: "80/20 split", description: "Standard split", value: "80/20 split" },
          { title: "70/30 split", description: "More validation data", value: "70/30 split" },
        ],
      }),
      collapsed: true,
    }]
    render(
      <MessageBubble
        message={makeMsg("Pick one:", toolResults)}
        onOptionSelect={onSelect}
      />,
      { wrapper: Wrapper },
    )
    const cards = screen.getAllByRole("button")
    expect(cards).toHaveLength(3) // 2 options + "Other"
    expect(cards[0]).toHaveTextContent("80/20 split")
    expect(cards[1]).toHaveTextContent("70/30 split")

    fireEvent.click(cards[0]!)
    expect(onSelect).toHaveBeenCalledWith("80/20 split", "1")
  })

  it("does not render option cards for user messages", () => {
    const userMsg: ChatMessage = {
      ...makeMsg("Some text"),
      role: "user",
    }
    render(<MessageBubble message={userMsg} onOptionSelect={vi.fn()} />, { wrapper: Wrapper })
    expect(screen.queryAllByRole("button")).toHaveLength(0)
  })

  it("does not render options when message has explicit optionCards", () => {
    const msg: ChatMessage = {
      ...makeMsg("Pick one:", [{
        name: "present_options",
        result: JSON.stringify({
          options: [
            { title: "A", description: "First", value: "a" },
          ],
        }),
        collapsed: true,
      }]),
      optionCards: [{ title: "Explicit", description: "Card", value: "explicit" }],
    }
    render(<MessageBubble message={msg} onOptionSelect={vi.fn()} />)
    expect(screen.getByText("Explicit")).toBeInTheDocument()
  })
})
