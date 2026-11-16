import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router"
import { useChatStore } from "@/stores/chat-store"
import { ChatInput } from "./chat-input"

function Wrapper({ children }: { children: React.ReactNode }) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return (
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    </MemoryRouter>
  )
}

const MESSAGE = "evaluate my vllm at https://v.example.com/v1 with key sk-proj-AbCdEf0123456789xyz"

function typeAndSend(text: string) {
  const textarea = screen.getByPlaceholderText("Type a message...") as HTMLTextAreaElement
  fireEvent.change(textarea, { target: { value: text } })
  fireEvent.keyDown(textarea, { key: "Enter" })
  return textarea
}

describe("ChatInput API key warning", () => {
  beforeEach(() => {
    useChatStore.setState({ currentConversationId: "test-conv", conversations: [], drafts: {} })
  })

  it("holds back a message that contains a key", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />, { wrapper: Wrapper })

    typeAndSend(MESSAGE)

    expect(onSend).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toHaveTextContent("looks like it contains an API key")
  })

  it("removes the key and keeps the rest of the message", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />, { wrapper: Wrapper })

    const textarea = typeAndSend(MESSAGE)
    fireEvent.click(screen.getByRole("button", { name: "Remove the key" }))

    expect(textarea.value).toBe("evaluate my vllm at https://v.example.com/v1 with key [key removed]")
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    fireEvent.keyDown(textarea, { key: "Enter" })
    expect(onSend).toHaveBeenCalledWith("evaluate my vllm at https://v.example.com/v1 with key [key removed]")
  })

  it("sends unchanged on Send anyway", () => {
    const onSend = vi.fn()
    render(<ChatInput onSend={onSend} />, { wrapper: Wrapper })

    typeAndSend(MESSAGE)
    fireEvent.click(screen.getByRole("button", { name: "Send anyway" }))

    expect(onSend).toHaveBeenCalledWith(MESSAGE)
  })
})
