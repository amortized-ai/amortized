import { createContext, useContext, useState, useCallback } from "react"
import { cn } from "@/lib/utils"

interface TabsContextValue {
  activeValue: string
  setActiveValue: (v: string) => void
}

const TabsContext = createContext<TabsContextValue>({
  activeValue: "",
  setActiveValue: () => {},
})

function Tabs({
  value,
  defaultValue,
  onValueChange,
  children,
  className,
}: {
  value?: string
  defaultValue?: string
  onValueChange?: (value: string) => void
  children: React.ReactNode
  className?: string
}) {
  const [internalValue, setInternalValue] = useState(defaultValue ?? "")
  const activeValue = value ?? internalValue

  const setActiveValue = useCallback(
    (v: string) => {
      if (value === undefined) setInternalValue(v)
      onValueChange?.(v)
    },
    [value, onValueChange],
  )

  return (
    <TabsContext.Provider value={{ activeValue, setActiveValue }}>
      <div className={cn("flex flex-col", className)}>{children}</div>
    </TabsContext.Provider>
  )
}

function TabsList({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "inline-flex h-9 items-center gap-1 border-b border-border",
        className,
      )}
    >
      {children}
    </div>
  )
}

function TabsTrigger({
  value,
  children,
  className,
}: {
  value: string
  children: React.ReactNode
  className?: string
  disabled?: boolean
}) {
  const { activeValue, setActiveValue } = useContext(TabsContext)
  const isActive = activeValue === value

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isActive}
      data-state={isActive ? "active" : "inactive"}
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap px-3 py-1.5 text-sm font-medium transition-colors",
        "border-b-[3px] -mb-px",
        isActive
          ? "border-[var(--rh-color-accent-base-on-light,#0066cc)] text-foreground dark:border-[var(--rh-color-accent-base-on-dark,#92c5f9)]"
          : "border-transparent text-muted-foreground hover:text-foreground hover:border-border",
        className,
      )}
      onClick={() => setActiveValue(value)}
    >
      {children}
    </button>
  )
}

function TabsContent({
  value,
  children,
  className,
}: {
  value: string
  children: React.ReactNode
  className?: string
}) {
  const { activeValue } = useContext(TabsContext)
  if (activeValue !== value) return null

  return (
    <div role="tabpanel" className={cn("mt-2", className)}>
      {children}
    </div>
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
