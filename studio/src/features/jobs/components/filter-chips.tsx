/* eslint-disable react-refresh/only-export-components */
import { cn } from "@/lib/utils"

interface ChipColor {
  idle: string
  hover: string
  active: string
}

interface FilterChipOption<T extends string> {
  value: T
  label: string
  color?: ChipColor
}

interface FilterChipsProps<T extends string> {
  label: string
  options: FilterChipOption<T>[]
  selected: T[]
  onChange: (selected: T[]) => void
}

const DEFAULT_CHIP: ChipColor = {
  idle: "border-border/60 text-muted-foreground",
  hover: "hover:border-border hover:bg-muted/60 hover:text-foreground",
  active: "border-foreground/20 bg-foreground/5 text-foreground shadow-sm",
}

export const TYPE_COLORS: Record<string, ChipColor> = {
  training: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-[#92c5f9] hover:bg-[#e0f0ff]/60 hover:text-[#004d99] dark:hover:border-[#003366] dark:hover:bg-[#003366]/30 dark:hover:text-[#4394e5]",
    active: "border-[#92c5f9] bg-[#e0f0ff] text-[#004d99] shadow-sm shadow-[#b9dafc]/50 dark:border-[#003366] dark:bg-[#003366]/40 dark:text-[#4394e5] dark:shadow-none",
  },
  sdg: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-[#d0c5f4] hover:bg-[#ece6ff]/60 hover:text-[#3d2785] dark:hover:border-[#21134d] dark:hover:bg-[#1b0d33]/30 dark:hover:text-[#876fd4]",
    active: "border-[#b6a6e9] bg-[#ece6ff] text-[#3d2785] shadow-sm shadow-[#d0c5f4]/50 dark:border-[#21134d] dark:bg-[#1b0d33]/40 dark:text-[#876fd4] dark:shadow-none",
  },
  upload: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-[#6ee7b7] hover:bg-[#d1fae5]/60 hover:text-[#065f46] dark:hover:border-[#064e3b] dark:hover:bg-[#064e3b]/30 dark:hover:text-[#34d399]",
    active: "border-[#6ee7b7] bg-[#d1fae5] text-[#065f46] shadow-sm shadow-[#a7f3d0]/50 dark:border-[#064e3b] dark:bg-[#064e3b]/40 dark:text-[#34d399] dark:shadow-none",
  },
}

export const STATUS_COLORS: Record<string, ChipColor> = {
  queued: DEFAULT_CHIP,
  provisioning: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-[#92c5f9] hover:bg-[#e0f0ff]/60 hover:text-[#004d99] dark:hover:border-[#003366] dark:hover:bg-[#003366]/30 dark:hover:text-[#4394e5]",
    active: "border-[#92c5f9] bg-[#e0f0ff] text-[#004d99] shadow-sm dark:border-[#003366] dark:bg-[#003366]/40 dark:text-[#4394e5]",
  },
  running: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-[#ffe072] hover:bg-[#fff4cc]/60 hover:text-[#96640f] dark:hover:border-[#73480b] dark:hover:bg-[#54330b]/30 dark:hover:text-[#dca614]",
    active: "border-[#ffe072] bg-[#fff4cc] text-[#96640f] shadow-sm dark:border-[#73480b] dark:bg-[#54330b]/40 dark:text-[#dca614]",
  },
  succeeded: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-[#afdc8f] hover:bg-[#e9f7df]/60 hover:text-[#204d00] dark:hover:border-[#204d00] dark:hover:bg-[#204d00]/30 dark:hover:text-[#63993d]",
    active: "border-[#87bb62] bg-[#e9f7df] text-[#204d00] shadow-sm dark:border-[#204d00] dark:bg-[#204d00]/40 dark:text-[#63993d]",
  },
  failed: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-rh-danger/30 hover:bg-rh-danger-light/60 hover:text-rh-danger-dark dark:hover:border-rh-danger-dark dark:hover:bg-rh-danger-dark/30 dark:hover:text-rh-danger",
    active: "border-rh-danger/40 bg-rh-danger-light/50 text-rh-danger-dark shadow-sm dark:border-rh-danger-dark dark:bg-rh-danger-dark/40 dark:text-rh-danger",
  },
  cancelled: {
    idle: "border-border/60 text-muted-foreground",
    hover: "hover:border-rh-yellow/30 hover:bg-rh-yellow-light/60 hover:text-rh-yellow-dark dark:hover:border-rh-yellow-dark dark:hover:bg-rh-yellow-dark/30 dark:hover:text-rh-yellow",
    active: "border-rh-yellow/30 bg-rh-yellow-light/80 text-rh-yellow-dark shadow-sm dark:border-rh-yellow-dark dark:bg-rh-yellow-dark/40 dark:text-rh-yellow",
  },
}

export function FilterChips<T extends string>({
  label,
  options,
  selected,
  onChange,
}: FilterChipsProps<T>) {
  function toggle(value: T) {
    if (selected.includes(value)) {
      onChange(selected.filter((v) => v !== value))
    } else {
      onChange([...selected, value])
    }
  }

  return (
    <div className="flex items-center gap-1.5">
      <span className="text-sm font-medium text-muted-foreground/70 mr-2">{label}</span>
      {options.map((opt) => {
        const isActive = selected.includes(opt.value)
        const color = opt.color ?? DEFAULT_CHIP

        return (
          <button
            key={opt.value}
            onClick={() => toggle(opt.value)}
            className={cn(
              "inline-flex items-center rounded-md border px-3 py-1.5 text-sm font-medium",
              "transition-all duration-200 ease-out",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50",
              isActive
                ? [color.active, "hover:brightness-95"]
                : [color.idle, "bg-transparent", color.hover],
            )}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}
