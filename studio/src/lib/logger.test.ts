import { describe, it, expect } from "vitest"
import { logger, getLogger } from "@/lib/logger"

describe("logger", () => {
  it("exports a root logger", () => {
    expect(logger).toBeDefined()
    expect(typeof logger.info).toBe("function")
    expect(typeof logger.error).toBe("function")
    expect(typeof logger.debug).toBe("function")
    expect(typeof logger.warn).toBe("function")
  })

  it("creates named sub-loggers via getLogger", () => {
    const child = getLogger("test-module")
    expect(child).toBeDefined()
    expect(typeof child.info).toBe("function")
    expect(typeof child.getSubLogger).toBe("function")
  })

  it("creates distinct sub-loggers for different names", () => {
    const a = getLogger("module-a")
    const b = getLogger("module-b")
    expect(a).not.toBe(b)
  })

  it("produces structured log output with attached metadata", () => {
    const child = getLogger("structured-test")
    const logs: unknown[] = []
    child.attachTransport((logObj) => {
      logs.push(logObj)
    })

    child.info("test message", { requestId: "abc-123", duration: 42 })

    expect(logs.length).toBe(1)
    const entry = logs[0] as Record<string, unknown>
    expect(entry["0"]).toBe("test message")
    const metadata = entry["1"] as Record<string, unknown>
    expect(metadata).toHaveProperty("requestId", "abc-123")
    expect(metadata).toHaveProperty("duration", 42)
    expect(entry).toHaveProperty("_meta")
    const meta = entry._meta as Record<string, unknown>
    expect(meta).toHaveProperty("name", "structured-test")
  })

  it("sub-loggers inherit root logger configuration", () => {
    const child = getLogger("inherit-test")
    const logs: unknown[] = []
    child.attachTransport((logObj) => {
      logs.push(logObj)
    })

    child.warn("warning with context", { module: "test" })

    expect(logs.length).toBe(1)
    const entry = logs[0] as Record<string, unknown>
    const metadata = entry["1"] as Record<string, unknown>
    expect(metadata).toHaveProperty("module", "test")
  })
})
