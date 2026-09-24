import { create } from "zustand"
import { persist } from "zustand/middleware"
import { getLogger } from "@/lib/logger"
import { DEFAULT_CHAT_MODEL_SELECTION } from "@/features/chat/models"

const logger = getLogger("settings-store")

interface SettingsState {
  apiKey: string
  chatModelSelection: string
  enabledProviders: string[]
  // Connected providers already auto-considered — so each is auto-enabled at most once and a
  // user's later disable is never undone. Persisted, so it survives reloads/remounts.
  observedConnectedProviders: string[]
  setApiKey: (key: string) => void
  setChatModelSelection: (selection: string) => void
  setEnabledProviders: (providers: string[]) => void
  toggleProvider: (providerID: string) => void
  enableNewlyConnected: (connected: string[]) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      apiKey: "",
      chatModelSelection: DEFAULT_CHAT_MODEL_SELECTION,
      enabledProviders: ["google-vertex-anthropic"],
      observedConnectedProviders: ["google-vertex-anthropic"],
      setApiKey: (_apiKey) => {
        logger.info("setApiKey", { masked: true })
        set({ apiKey: _apiKey })
      },
      setChatModelSelection: (chatModelSelection) => {
        logger.info("setChatModelSelection", { chatModelSelection })
        set({ chatModelSelection })
      },
      setEnabledProviders: (enabledProviders) => {
        logger.info("setEnabledProviders", { enabledProviders })
        set({ enabledProviders })
      },
      toggleProvider: (providerID) => {
        const current = get().enabledProviders
        const next = current.includes(providerID)
          ? current.filter((id) => id !== providerID)
          : [...current, providerID]
        logger.info("toggleProvider", { providerID, enabled: next.includes(providerID) })
        set({ enabledProviders: next })
      },
      enableNewlyConnected: (connected) => {
        // Enable a provider only the FIRST time it's seen connected (not in the observed
        // set), then record it as observed. A provider already observed is left untouched,
        // so a user's disable is never re-enabled. Reads via get() so it's not driven by a
        // stale render closure.
        const { enabledProviders, observedConnectedProviders } = get()
        const newly = connected.filter((id) => !observedConnectedProviders.includes(id))
        if (newly.length === 0) return
        const toEnable = newly.filter((id) => !enabledProviders.includes(id))
        logger.info("enableNewlyConnected", { newly })
        set({
          enabledProviders: [...enabledProviders, ...toEnable],
          observedConnectedProviders: [...observedConnectedProviders, ...newly],
        })
      },
    }),
    {
      name: "amortized-settings",
      version: 5,
      migrate: (persisted: unknown, version: number) => {
        const state = persisted as Record<string, unknown>

        if (version < 3) {
          state.chatModelSelection = DEFAULT_CHAT_MODEL_SELECTION
          delete state.chatModel
          delete state.chatModelID
        }

        if (version < 4) {
          state.enabledProviders = ["google-vertex-anthropic"]
          state.chatModelSelection = DEFAULT_CHAT_MODEL_SELECTION
        }

        if (version < 5) {
          // Seed the observed-connected set from the current enabled set: existing choices
          // are respected, and any connected-but-not-enabled provider is auto-enabled once on
          // the next load, after which disables stick.
          state.observedConnectedProviders = Array.isArray(state.enabledProviders)
            ? state.enabledProviders
            : ["google-vertex-anthropic"]
        }

        return state as unknown as SettingsState
      },
    },
  ),
)

export function maskApiKey(key: string): string {
  if (key.length <= 4) return "****"
  return `${"*".repeat(key.length - 4)}${key.slice(-4)}`
}
