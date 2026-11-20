import { describe, it, expect } from "vitest"
import { render, screen } from "@testing-library/react"
import { ModelCheckWarnings } from "./model-check-warnings"
import type { ModelCheckResult } from "@/types/api"

const check = (overrides: Partial<ModelCheckResult>): ModelCheckResult => ({
  label: "judge",
  model: "gpt-oss",
  base_url: "http://gw/v1",
  outcome: "unverified",
  reason: "deadline",
  detail: "",
  ...overrides,
})

describe("ModelCheckWarnings", () => {
  it("explains why an endpoint wasn't verified, without blocking language", () => {
    render(<ModelCheckWarnings checks={[check({})]} />)
    expect(
      screen.getByText("Judge gpt-oss not verified: it didn't answer in time (a cold endpoint may still serve the job)."),
    ).toBeInTheDocument()
  })

  it("renders only unverified results, one row each (even with a repeated label and model)", () => {
    render(
      <ModelCheckWarnings
        checks={[
          check({ outcome: "passed", reason: "ok" }),
          check({ reason: "server_error" }),
          check({ reason: "unreachable" }),
        ]}
      />,
    )
    expect(screen.getAllByText(/not verified/)).toHaveLength(2)
  })

  it("renders nothing when every check passed", () => {
    const { container } = render(<ModelCheckWarnings checks={[check({ outcome: "passed", reason: "ok" })]} />)
    expect(container).toBeEmptyDOMElement()
  })
})
