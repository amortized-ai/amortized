import { describe, it, expect } from "vitest"
import { collectMissingRequired, collectRequiredPaths } from "./recipe-validation"

describe("collectRequiredPaths (branch-aware markers)", () => {
  // Mirrors the real schema: tool_alias is required in one $def but OPTIONAL in the
  // LLM column branch. A name-keyed union would wrongly mark it required everywhere.
  const schema = {
    required: ["columns"],
    properties: { columns: { type: "array", items: { $ref: "#/$defs/Column" } } },
    $defs: {
      Column: {
        oneOf: [{ $ref: "#/$defs/LLMText" }, { $ref: "#/$defs/Tool" }],
        discriminator: { propertyName: "column_type" },
      },
      LLMText: {
        type: "object",
        required: ["column_type", "name", "model_alias"],
        properties: {
          column_type: { const: "llm-text" },
          name: { type: "string" },
          model_alias: { type: "string" },
          tool_alias: { type: "string" }, // optional here
        },
      },
      Tool: {
        type: "object",
        required: ["column_type", "tool_alias"], // required here
        properties: { column_type: { const: "tool" }, tool_alias: { type: "string" } },
      },
    },
  }

  it("marks a field required only in the branch that requires it", () => {
    const config = {
      columns: [{ column_type: "llm-text", name: "q", model_alias: "text", tool_alias: "" }],
    }
    const paths = collectRequiredPaths(config, schema)
    expect(paths.has("columns/0/model_alias")).toBe(true) // required in the llm-text branch
    expect(paths.has("columns/0/tool_alias")).toBe(false) // optional in llm-text (no false positive)
    expect(paths.has("columns")).toBe(true) // top-level required
  })

  it("includes per-recipe requiredKeys as top-level paths", () => {
    expect(collectRequiredPaths({ document_ids: [] }, null, ["document_ids"]).has("document_ids")).toBe(
      true,
    )
  })
})

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
      ["Model Configs › Model", "Model Configs › Provider"].sort(),
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
    expect(collectMissingRequired(config, schema)).toEqual(["Columns › Model Alias"])
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
