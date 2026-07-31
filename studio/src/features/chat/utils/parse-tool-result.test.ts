import { describe, it, expect } from "vitest"
import { unwrapToolResult, extractJobInfo } from "./parse-tool-result"

describe("unwrapToolResult", () => {
  it("parses a plain JSON object string", () => {
    const result = unwrapToolResult(JSON.stringify({ id: "abc-123", type: "training" }))
    expect(result).toEqual({ id: "abc-123", type: "training" })
  })

  it("unwraps MCP content block array", () => {
    const mcp = JSON.stringify([{ type: "text", text: JSON.stringify({ id: "abc-123", type: "sdg" }) }])
    const result = unwrapToolResult(mcp)
    expect(result).toEqual({ id: "abc-123", type: "sdg" })
  })

  it("returns null for empty string", () => {
    expect(unwrapToolResult("")).toBeNull()
  })

  it("returns null for invalid JSON", () => {
    expect(unwrapToolResult("not json")).toBeNull()
  })

  it("returns null for MCP array with no text block", () => {
    const mcp = JSON.stringify([{ type: "image", data: "..." }])
    expect(unwrapToolResult(mcp)).toBeNull()
  })

  it("returns null for MCP array with unparseable text", () => {
    const mcp = JSON.stringify([{ type: "text", text: "not json" }])
    expect(unwrapToolResult(mcp)).toBeNull()
  })
})

describe("extractJobInfo", () => {
  const JOB_UUID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

  it("extracts id and type from plain JSON object", () => {
    const result = extractJobInfo(JSON.stringify({ id: JOB_UUID, type: "training" }))
    expect(result).toEqual({ jobId: JOB_UUID, jobType: "TRAINING" })
  })

  it("defaults type to SDG when not present", () => {
    const result = extractJobInfo(JSON.stringify({ id: JOB_UUID }))
    expect(result).toEqual({ jobId: JOB_UUID, jobType: "SDG" })
  })

  it("extracts from MCP content block array", () => {
    const mcp = JSON.stringify([{ type: "text", text: JSON.stringify({ id: JOB_UUID, type: "sdg" }) }])
    const result = extractJobInfo(mcp)
    expect(result).toEqual({ jobId: JOB_UUID, jobType: "SDG" })
  })

  it("falls back to UUID regex scan", () => {
    const result = extractJobInfo(`Job created: ${JOB_UUID} successfully`)
    expect(result).toEqual({ jobId: JOB_UUID, jobType: "SDG" })
  })

  it("skips dry_run results", () => {
    const result = extractJobInfo(JSON.stringify({ dry_run: true, valid: true, type: "sdg", config: {} }))
    expect(result).toEqual({ jobId: null, jobType: "SDG" })
  })

  it("returns null jobId for empty string", () => {
    expect(extractJobInfo("")).toEqual({ jobId: null, jobType: "SDG" })
  })

  it("returns null jobId when no UUID found", () => {
    expect(extractJobInfo("some random text")).toEqual({ jobId: null, jobType: "SDG" })
  })
})
