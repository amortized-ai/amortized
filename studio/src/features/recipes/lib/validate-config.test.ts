import { describe, it, expect } from "vitest"
import { missingRequiredFields } from "./validate-config"

describe("missingRequiredFields", () => {
  const schema = { required: ["columns", "topic"], properties: {} }

  it("flags missing or empty required fields", () => {
    expect(missingRequiredFields({ topic: "x" }, schema)).toEqual(["columns"])
    expect(missingRequiredFields({ columns: [], topic: "" }, schema)).toEqual([
      "columns",
      "topic",
    ])
    expect(missingRequiredFields({ columns: [{ name: "a" }], topic: "   " }, schema)).toEqual([
      "topic",
    ])
  })

  it("passes when required fields are present and non-empty", () => {
    expect(
      missingRequiredFields({ columns: [{ name: "a" }], topic: "qa" }, schema),
    ).toEqual([])
  })

  it("returns [] when there is no schema or no required list", () => {
    expect(missingRequiredFields({}, undefined)).toEqual([])
    expect(missingRequiredFields({}, null)).toEqual([])
    expect(missingRequiredFields({}, { properties: {} })).toEqual([])
  })
})
