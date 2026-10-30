import { describe, it, expect } from "vitest"
import {
  stripPlaceholders,
  isPromptRef,
  makePromptRef,
  promptRefId,
  findPromptRefs,
  resolvePromptRefs,
} from "./placeholders"

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

  it("does not strip prompt refs (they are valid values)", () => {
    expect(
      stripPlaceholders({ columns: [{ system_prompt: makePromptRef("prm_a") }] }),
    ).toEqual({ columns: [{ system_prompt: "<prompt:prm_a>" }] })
  })
})

describe("prompt refs", () => {
  it("recognizes only a full-value ref token", () => {
    expect(isPromptRef(makePromptRef("prm_a"))).toBe(true)
    expect(promptRefId("<prompt:prm_a>")).toBe("prm_a")
    expect(isPromptRef("see <prompt:prm_a>")).toBe(false)
    expect(isPromptRef("<prompt:nope>")).toBe(false)
    expect(promptRefId(123)).toBe(null)
  })

  it("finds refs across the config tree", () => {
    const config = {
      columns: [
        { system_prompt: makePromptRef("prm_a") },
        { system_prompt: "inline" },
      ],
      processors: [{ template: { messages: [{ content: makePromptRef("prm_b") }] } }],
    }
    expect(findPromptRefs(config).sort()).toEqual(["prm_a", "prm_b"])
  })

  it("resolves refs to text purely (source untouched, unknown left as-is)", () => {
    const config = {
      columns: [{ system_prompt: makePromptRef("prm_a") }, { system_prompt: makePromptRef("prm_x") }],
    }
    const resolved = resolvePromptRefs(config, { prm_a: "FULL TEXT" })
    expect(resolved.columns[0]?.system_prompt).toBe("FULL TEXT")
    expect(resolved.columns[1]?.system_prompt).toBe("<prompt:prm_x>")
    expect(config.columns[0]?.system_prompt).toBe("<prompt:prm_a>")
  })
})
