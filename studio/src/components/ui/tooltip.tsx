import { Children, cloneElement, isValidElement } from "react"

function TooltipProvider({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}

function Tooltip({ children }: { children: React.ReactNode }) {
  let triggerChild: React.ReactElement | null = null
  let contentText = ""

  Children.forEach(children, (child) => {
    if (isValidElement(child)) {
      if (child.type === TooltipTrigger) {
        triggerChild = child
      } else if (child.type === TooltipContent) {
        const props = child.props as { children?: React.ReactNode }
        contentText = typeof props.children === "string" ? props.children : ""
      }
    }
  })

  if (triggerChild) {
    const triggerProps = (triggerChild as React.ReactElement<{ children?: React.ReactNode }>).props
    const innerChild = triggerProps.children

    if (isValidElement(innerChild)) {
      return cloneElement(innerChild as React.ReactElement<{ title?: string }>, {
        title: contentText,
      })
    }
  }

  return <>{triggerChild}</>
}

function TooltipTrigger({
  children,
}: {
  children: React.ReactNode
  asChild?: boolean
} & React.HTMLAttributes<HTMLElement>) {
  return <>{children}</>
}

function TooltipContent({
  children,
}: {
  children: React.ReactNode
  side?: string
  className?: string
  sideOffset?: number
}) {
  void children
  return null
}

export { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider }
