import { createContext, useContext, useState, useCallback } from "react"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

interface AccordionContextValue {
  expandedValues: Set<string>
  toggle: (value: string) => void
  type: "single" | "multiple"
}

const AccordionContext = createContext<AccordionContextValue>({
  expandedValues: new Set(),
  toggle: () => {},
  type: "single",
})

function Accordion({
  type = "single",
  defaultValue,
  children,
  className,
}: {
  type?: "single" | "multiple"
  defaultValue?: string | string[]
  children: React.ReactNode
  className?: string
}) {
  const [expandedValues, setExpandedValues] = useState<Set<string>>(() => {
    if (!defaultValue) return new Set()
    return new Set(Array.isArray(defaultValue) ? defaultValue : [defaultValue])
  })

  const toggle = useCallback(
    (value: string) => {
      setExpandedValues((prev) => {
        const next = new Set(prev)
        if (next.has(value)) {
          next.delete(value)
        } else {
          if (type === "single") next.clear()
          next.add(value)
        }
        return next
      })
    },
    [type],
  )

  return (
    <AccordionContext.Provider value={{ expandedValues, toggle, type }}>
      <div className={className}>{children}</div>
    </AccordionContext.Provider>
  )
}

function AccordionItem({
  value,
  children,
  className,
}: {
  value: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      data-slot="accordion-item"
      data-value={value}
      className={cn(
        "border-b border-[var(--rh-border-width-sm,1px)] border-border",
        className,
      )}
    >
      {children}
    </div>
  )
}

function AccordionTrigger({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  const { expandedValues, toggle } = useContext(AccordionContext)
  const item = useContext(AccordionItemContext)
  const isOpen = expandedValues.has(item)

  return (
    <h3>
      <button
        type="button"
        aria-expanded={isOpen}
        data-state={isOpen ? "open" : "closed"}
        className={cn(
          "flex w-full flex-1 items-center justify-between py-4 text-sm font-medium transition-all hover:underline text-left [&[data-state=open]>svg]:rotate-180",
          className,
        )}
        onClick={() => toggle(item)}
      >
        {children}
        <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200" />
      </button>
    </h3>
  )
}

const AccordionItemContext = createContext<string>("")

function AccordionItemWrapper({
  value,
  children,
  className,
}: {
  value: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <AccordionItemContext.Provider value={value}>
      <AccordionItem value={value} className={className}>
        {children}
      </AccordionItem>
    </AccordionItemContext.Provider>
  )
}

function AccordionContent({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  const { expandedValues } = useContext(AccordionContext)
  const item = useContext(AccordionItemContext)
  const isOpen = expandedValues.has(item)

  if (!isOpen) return null

  return (
    <div
      role="region"
      data-state={isOpen ? "open" : "closed"}
      className="overflow-hidden text-sm"
    >
      <div className={cn("pb-4 pt-0", className)}>{children}</div>
    </div>
  )
}

export {
  Accordion,
  AccordionItemWrapper as AccordionItem,
  AccordionTrigger,
  AccordionContent,
}
