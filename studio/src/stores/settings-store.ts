import { create } from "zustand"
import { persist } from "zustand/middleware"
import { getLogger } from "@/lib/logger"
import { DEFAULT_CHAT_MODEL_SELECTION } from "@/features/chat/models"

const logger = getLogger("settings-store")

interface SettingsState {
  apiKey: string
  chatModelSelection: string
  enabledProviders: string[]
  setApiKey: (key: string) => void
  setChatModelSelection: (selection: string) => void
  setEnabledProviders: (providers: string[]) => void
  toggleProvider: (providerID: string) => void
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      apiKey: "",
      chatModelSelection: DEFAULT_CHAT_MODEL_SELECTION,
      enabledProviders: ["google-vertex-anthropic"],
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
    }),
    {
      name: "amortized-settings",
      version: 4,
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

        return state as unknown as SettingsState
      },
    },
  ),
)

export function maskApiKey(key: string): string {
  if (key.length <= 4) return "****"
  return `${"*".repeat(key.length - 4)}${key.slice(-4)}`
}
