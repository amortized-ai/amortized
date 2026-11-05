import { describe, it, expect } from "vitest"
import { jobTypeLabel } from "./format"

describe("jobTypeLabel", () => {
  it("labels known job types, case-insensitively", () => {
    expect(jobTypeLabel("sdg")).toBe("SDG")
    expect(jobTypeLabel("eval")).toBe("Eval")
    expect(jobTypeLabel("TRAINING")).toBe("Training")
  })

  it("falls back to Job for an unknown or missing type", () => {
    expect(jobTypeLabel("mystery")).toBe("Job")
    expect(jobTypeLabel("")).toBe("Job")
    expect(jobTypeLabel(undefined)).toBe("Job")
  })
})
