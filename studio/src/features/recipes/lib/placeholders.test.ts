import { describe, it, expect } from "vitest"
import { stripPlaceholders } from "./placeholders"

describe("stripPlaceholders", () => {
  it("drops placeholder entries from arrays (document_ids)", () => {
    expect(stripPlaceholders({ document_ids: ["<doc-id-from-list_documents>"] })).toEqual({
      document_ids: [],
    })
  })

  it("empties placeholder strings but keeps the field (model/provider)", () => {
    expect(
      stripPlaceholders({
        model_configs: [
          { alias: "text", model: "<from-list_models>", provider: "<from-list_models>" },
        ],
      }),
    ).toEqual({
      model_configs: [{ alias: "text", model: "", provider: "" }],
    })
  })

  it("empties a placeholder parent_job_id", () => {
    expect(stripPlaceholders({ parent_job_id: "<SDG_JOB_ID>" })).toEqual({ parent_job_id: "" })
  })

  it("leaves real values untouched", () => {
    const config = {
      num_records: 500,
      document_ids: ["abc123"],
      model_configs: [{ alias: "text", model: "gpt-oss", provider: "gateway" }],
    }
    expect(stripPlaceholders(config)).toEqual(config)
  })
})
