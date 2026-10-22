import { Label } from "@/components/ui/label"
import { formatLabel } from "./recipe-field-utils"

// Shared field primitives used by both the generic recipe renderer
// (recipe-config-form.tsx) and the data-wired pickers (recipe-pickers.tsx).
// Kept in their own module so the pickers can reuse the required-marker label
// without a circular import.

export function FieldLabel({
  label,
  htmlFor,
  required,
}: {
  label: string
  htmlFor?: string
  required?: boolean
}) {
  return (
    <Label htmlFor={htmlFor} className="text-sm font-medium">
      {formatLabel(label)}
      {required && (
        <span className="ml-0.5 text-red-500" aria-label="required">
          *
        </span>
      )}
    </Label>
  )
}

export function FieldDescription({ text }: { text?: string }) {
  if (!text) return null
  return <p className="text-xs text-muted-foreground">{text}</p>
}
