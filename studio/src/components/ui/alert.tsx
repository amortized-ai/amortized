type AlertVariant = "default" | "destructive"
type RhAlertState = "info" | "danger"

const variantToState: Record<AlertVariant, RhAlertState> = {
  default: "info",
  destructive: "danger",
}

function Alert({
  variant = "default",
  children,
  className,
  ...props
}: {
  variant?: AlertVariant
  children?: React.ReactNode
  className?: string
} & React.HTMLAttributes<HTMLElement>) {
  return (
    <rh-alert state={variantToState[variant]} className={className} {...props}>
      {children}
    </rh-alert>
  )
}

function AlertTitle({ children }: { children?: React.ReactNode }) {
  return <h4 slot="header">{children}</h4>
}

function AlertDescription({ children }: { children?: React.ReactNode }) {
  return <>{children}</>
}

export { Alert, AlertTitle, AlertDescription }
