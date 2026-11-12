import { describe, expect, it } from "vitest"
import { findApiKeys, redactApiKeys } from "./detect-api-key"

describe("findApiKeys", () => {
  it.each([
    "evaluate my vllm, key sk-proj-AbCdEf0123456789xyzXYZ",
    "use sk-ant-api03-abcdefghijklmnop1234 as the judge key",
    "token: eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ1c2VyMTIzIn0.c2lnbmF0dXJlMTIzNDU2",
    "here is my hf token hf_abcdefghijklmnopqrstuvwxyz123456",
    "api key = 7f3a9c2e1b8d4f6a0e5c",
    "Authorization: Bearer abc123def456ghi789",
    "the password is Sup3rSecretValue123",
  ])("flags %s", (text) => {
    expect(findApiKeys(text)).not.toHaveLength(0)
  })

  it.each([
    "evaluate the model from job 5298f363-1c2d-4e5f-8a9b-0c1d2e3f4a5b",
    "set max_tokens to 4096 and use the prompt token <prompt:prm_abc123def456>",
    "what are the key metrics for a classifier?",
    "risk-assessment-model-2026-v1-final is my model name",
    "my api key is already in settings",
    "use https://vllm.models.svc.cluster.local:8080/v1 for the endpoint",
  ])("does not flag %s", (text) => {
    expect(findApiKeys(text)).toHaveLength(0)
  })
})

describe("redactApiKeys", () => {
  it("replaces only the key and keeps the rest of the message", () => {
    expect(redactApiKeys("my vllm at https://v.example.com/v1, key sk-proj-AbCdEf0123456789xyz, model qwen")).toBe(
      "my vllm at https://v.example.com/v1, key [key removed], model qwen",
    )
  })

  it("merges overlapping matches", () => {
    expect(redactApiKeys("api key: sk-ant-api03-abcdefghijklmnop1234")).toBe("api key: [key removed]")
  })
})
