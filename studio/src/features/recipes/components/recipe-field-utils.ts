import { createContext } from "react"

// Non-component helpers shared across the recipe form modules. Kept out of the
// .tsx files so react-refresh's "only export components" rule stays happy.

export type FieldMeta = {
  description?: string
  enumValues?: string[]
  required?: boolean
}

export function formatLabel(label: string): string {
  return label.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

// Aliases declared in the current SDG config's `model_configs`, provided by
// SdgConfigForm so a column's `model_alias` can be picked from them (a
// model_alias must match one of these, not a raw catalog model id).
export const SdgModelAliasesContext = createContext<string[]>([])
