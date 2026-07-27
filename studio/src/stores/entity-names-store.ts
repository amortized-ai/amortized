import { create } from "zustand"
import { persist } from "zustand/middleware"

interface EntityNamesState {
  names: Record<string, string>
  setName: (id: string, name: string) => void
  getName: (id: string) => string | undefined
}

export const useEntityNamesStore = create<EntityNamesState>()(
  persist(
    (set, get) => ({
      names: {},
      setName: (id, name) =>
        set((s) => ({ names: { ...s.names, [id]: name } })),
      getName: (id) => get().names[id],
    }),
    { name: "amortized-entity-names" },
  ),
)
