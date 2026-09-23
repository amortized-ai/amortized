import { create } from "zustand"
import { persist } from "zustand/middleware"

type Theme = "light" | "dark"

interface UIState {
  sidebarCollapsed: boolean
  navSidebarWidth: number
  conversationsPanelWidth: number
  theme: Theme
  tutorialCompleted: boolean
  tutorialActive: boolean
  tutorialStep: number
  setSidebarCollapsed: (collapsed: boolean) => void
  setNavSidebarWidth: (width: number) => void
  setConversationsPanelWidth: (width: number) => void
  toggleSidebar: () => void
  setTheme: (theme: Theme) => void
  startTutorial: () => void
  nextTutorialStep: () => void
  prevTutorialStep: () => void
  goToTutorialStep: (step: number) => void
  completeTutorial: () => void
  skipTutorial: () => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      navSidebarWidth: 208,
      conversationsPanelWidth: 260,
      theme: (typeof window !== "undefined" && window.matchMedia?.("(prefers-color-scheme: dark)").matches) ? "dark" : "light",
      tutorialCompleted: false,
      tutorialActive: false,
      tutorialStep: 0,
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
      setNavSidebarWidth: (width) => set({ navSidebarWidth: width }),
      setConversationsPanelWidth: (width) => set({ conversationsPanelWidth: width }),
      toggleSidebar: () =>
        set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      setTheme: (theme) => set({ theme }),
      startTutorial: () =>
        set({ tutorialActive: true, tutorialStep: 0 }),
      nextTutorialStep: () =>
        set((state) => ({ tutorialStep: state.tutorialStep + 1 })),
      prevTutorialStep: () =>
        set((state) => ({
          tutorialStep: Math.max(0, state.tutorialStep - 1),
        })),
      goToTutorialStep: (step) => set({ tutorialStep: step }),
      completeTutorial: () =>
        set({ tutorialActive: false, tutorialCompleted: true, tutorialStep: 0 }),
      skipTutorial: () =>
        set({ tutorialActive: false, tutorialCompleted: true, tutorialStep: 0 }),
    }),
    {
      name: "amortized-ui",
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        navSidebarWidth: state.navSidebarWidth,
        conversationsPanelWidth: state.conversationsPanelWidth,
        theme: state.theme,
        tutorialCompleted: state.tutorialCompleted,
      }),
    },
  ),
)
