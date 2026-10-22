import { describe, it, expect } from "vitest"
import { collectMissingRequired } from "./recipe-validation"

describe("collectMissingRequired", () => {
  it("returns [] when no schema is provided", () => {
    expect(collectMissingRequired({ columns: [] })).toEqual([])
  })

  it("flags an empty top-level required array (columns)", () => {
    const schema = {
      required: ["columns"],
      properties: { columns: { type: "array", items: {} } },
    }
    expect(collectMissingRequired({ columns: [] }, schema)).toEqual(["Columns"])
    expect(collectMissingRequired({ columns: [{ name: "x" }] }, schema)).toEqual([])
  })

  it("flags empty nested required fields via $ref (model/provider)", () => {
    const schema = {
      required: [],
      properties: {
        model_configs: { type: "array", items: { $ref: "#/$defs/ModelConfig" } },
      },
      $defs: {
        ModelConfig: {
          type: "object",
          required: ["alias", "model", "provider"],
          properties: {
            alias: { type: "string" },
            model: { type: "string" },
            provider: { type: "string" },
          },
        },
      },
    }
    const config = { model_configs: [{ alias: "text", model: "", provider: "" }] }
    expect(collectMissingRequired(config, schema).sort()).toEqual(
      ["Model Configs[0] › Model", "Model Configs[0] › Provider"].sort(),
    )
  })

  it("selects the discriminated-union branch and flags its missing required", () => {
    const schema = {
      required: ["columns"],
      properties: { columns: { type: "array", items: { $ref: "#/$defs/Column" } } },
      $defs: {
        Column: {
          oneOf: [{ $ref: "#/$defs/Sampler" }, { $ref: "#/$defs/LLMText" }],
          discriminator: { propertyName: "column_type" },
        },
        Sampler: {
          type: "object",
          required: ["column_type", "name"],
          properties: { column_type: { const: "sampler" }, name: { type: "string" } },
        },
        LLMText: {
          type: "object",
          required: ["column_type", "name", "model_alias"],
          properties: {
            column_type: { const: "llm-text" },
            name: { type: "string" },
            model_alias: { type: "string" },
          },
        },
      },
    }
    const config = { columns: [{ column_type: "llm-text", name: "q", model_alias: "" }] }
    expect(collectMissingRequired(config, schema)).toEqual(["Columns[0] › Model Alias"])
  })

  it("flags an empty per-recipe requiredKey (document_ids)", () => {
    // document_ids is optional in the schema but required for this recipe.
    expect(collectMissingRequired({ document_ids: [] }, null, ["document_ids"])).toEqual([
      "Document Ids",
    ])
    expect(collectMissingRequired({ document_ids: ["abc123"] }, null, ["document_ids"])).toEqual([])
  })

  it("returns [] when everything required is filled", () => {
    const schema = {
      required: ["columns"],
      properties: {
        columns: { type: "array", items: { $ref: "#/$defs/Column" } },
        model_configs: { type: "array", items: { $ref: "#/$defs/ModelConfig" } },
      },
      $defs: {
        Column: {
          type: "object",
          required: ["name"],
          properties: { name: { type: "string" } },
        },
        ModelConfig: {
          type: "object",
          required: ["alias", "model", "provider"],
          properties: {
            alias: { type: "string" },
            model: { type: "string" },
            provider: { type: "string" },
          },
        },
      },
    }
    const config = {
      columns: [{ name: "answer" }],
      model_configs: [{ alias: "text", model: "gpt-oss", provider: "gateway" }],
    }
    expect(collectMissingRequired(config, schema)).toEqual([])
  })
})
