import { describe, it, expect, beforeEach } from "vitest"
import { useSettingsStore } from "./settings-store"

describe("settings-store — enableNewlyConnected", () => {
  beforeEach(() => {
    useSettingsStore.setState({
      enabledProviders: ["google-vertex-anthropic"],
      observedConnectedProviders: ["google-vertex-anthropic"],
    })
  })

  it("enables a newly-connected provider once and records it as observed", () => {
    useSettingsStore.getState().enableNewlyConnected(["google-vertex-anthropic", "openai"])
    const s = useSettingsStore.getState()
    expect(s.enabledProviders).toContain("openai")
    expect(s.observedConnectedProviders).toContain("openai")
  })

  it("does NOT re-enable a connected provider the user has disabled", () => {
    // openai connects for the first time -> auto-enabled + observed
    useSettingsStore.getState().enableNewlyConnected(["openai"])
    expect(useSettingsStore.getState().enabledProviders).toContain("openai")
    // user disables it
    useSettingsStore.getState().toggleProvider("openai")
    expect(useSettingsStore.getState().enabledProviders).not.toContain("openai")
    // still connected on a later pass -> must stay disabled (already observed)
    useSettingsStore.getState().enableNewlyConnected(["openai"])
    expect(useSettingsStore.getState().enabledProviders).not.toContain("openai")
  })

  it("is a no-op when every connected provider is already observed", () => {
    const before = useSettingsStore.getState().enabledProviders
    useSettingsStore.getState().enableNewlyConnected(["google-vertex-anthropic"])
    expect(useSettingsStore.getState().enabledProviders).toEqual(before)
  })
})
