import { create } from "zustand"
import { persist } from "zustand/middleware"

type Theme = "light" | "dark"

interface UIState {
  sidebarCollapsed: boolean
  navSidebarWidth: number
  conversationsPanelWidth: number
  theme: Theme
  setSidebarCollapsed: (collapsed: boolean) => void
  setNavSidebarWidth: (width: number) => void
  setConversationsPanelWidth: (width: number) => void
  toggleSidebar: () => void
  setTheme: (theme: Theme) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      navSidebarWidth: 208,
      conversationsPanelWidth: 260,
      theme: (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) ? "dark" : "light",
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setNavSidebarWidth: (width) => set({ navSidebarWidth: width }),
      setConversationsPanelWidth: (width) => set({ conversationsPanelWidth: width }),
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: "amortized-ui",
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        navSidebarWidth: state.navSidebarWidth,
        conversationsPanelWidth: state.conversationsPanelWidth,
        theme: state.theme,
      }),
    },
  ),
)
