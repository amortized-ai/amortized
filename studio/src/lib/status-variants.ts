export const statusVariants = {
  success: "bg-rh-green-light text-rh-green-dark border-rh-green/30 dark:bg-rh-green-dark/30 dark:text-rh-green dark:border-rh-green-dark",
  error: "bg-rh-danger-light text-rh-danger-dark border-rh-danger/30 dark:bg-rh-danger-dark/30 dark:text-rh-danger dark:border-rh-danger-dark",
  warning: "bg-rh-yellow-light text-rh-yellow-dark border-rh-yellow/30 dark:bg-rh-yellow-dark/30 dark:text-rh-yellow dark:border-rh-yellow-dark",
  info: "bg-rh-blue-light text-rh-blue-dark border-rh-blue/30 dark:bg-rh-blue-dark/30 dark:text-rh-blue dark:border-rh-blue-dark",
  neutral: "bg-muted text-muted-foreground border-border",
  running: "bg-rh-blue-light text-rh-blue-dark border-rh-blue/30 dark:bg-rh-blue-dark/30 dark:text-rh-blue dark:border-rh-blue-dark",
  queued: "bg-muted text-muted-foreground border-border",
  completed: "bg-rh-green-light text-rh-green-dark border-rh-green/30 dark:bg-rh-green-dark/30 dark:text-rh-green dark:border-rh-green-dark",
  failed: "bg-rh-danger-light text-rh-danger-dark border-rh-danger/30 dark:bg-rh-danger-dark/30 dark:text-rh-danger dark:border-rh-danger-dark",
  cancelled: "bg-muted text-muted-foreground border-border",
} as const

export type StatusVariant = keyof typeof statusVariants

export const scoreVariants = {
  best: "bg-rh-green-light/50 text-rh-green-dark dark:bg-rh-green-dark/30 dark:text-rh-green",
  worst: "bg-rh-danger-light/50 text-rh-danger-dark dark:bg-rh-danger-dark/30 dark:text-rh-danger",
} as const

export const roleVariants: Record<string, string> = {
  system: "bg-rh-purple-light text-rh-purple-dark border-rh-purple/30 dark:bg-rh-purple-dark/30 dark:text-rh-purple dark:border-rh-purple-dark",
  user: "bg-rh-blue-light text-rh-blue-dark border-rh-blue/30 dark:bg-rh-blue-dark/30 dark:text-rh-blue dark:border-rh-blue-dark",
  assistant: "bg-rh-green-light text-rh-green-dark border-rh-green/30 dark:bg-rh-green-dark/30 dark:text-rh-green dark:border-rh-green-dark",
}

export const fitsVariants = {
  fits: { icon: "text-rh-green dark:text-rh-green", text: "text-rh-green-dark dark:text-rh-green" },
  noFit: { icon: "text-rh-danger dark:text-rh-danger", text: "text-rh-danger-dark dark:text-rh-danger" },
} as const

export const statusIconVariants = {
  ok: "text-rh-green dark:text-rh-green",
  warning: "text-rh-yellow dark:text-rh-yellow",
  error: "text-rh-danger dark:text-rh-danger",
} as const

export const passRateVariants = {
  high: "text-rh-green dark:text-rh-green",
  medium: "text-rh-yellow dark:text-rh-yellow",
  low: "text-rh-danger dark:text-rh-danger",
} as const

export function getPassRateVariant(rate: number): string {
  if (rate >= 0.8) return passRateVariants.high
  if (rate >= 0.5) return passRateVariants.medium
  return passRateVariants.low
}
