import { render, screen } from "@testing-library/react"
import { describe, it, expect } from "vitest"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { MemoryRouter } from "react-router"
import type { ReactNode } from "react"
import { UploadDocumentDialog } from "./upload-document-dialog"

function renderWithProviders(ui: ReactNode) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  )
}

describe("UploadDocumentDialog", () => {
  it("renders its own trigger by default (Documents page usage)", () => {
    renderWithProviders(<UploadDocumentDialog />)
    expect(screen.getByTestId("upload-document")).toBeInTheDocument()
  })

  it("hides the trigger when showTrigger is false (in-chat usage)", () => {
    renderWithProviders(<UploadDocumentDialog showTrigger={false} />)
    expect(screen.queryByTestId("upload-document")).not.toBeInTheDocument()
  })
})
