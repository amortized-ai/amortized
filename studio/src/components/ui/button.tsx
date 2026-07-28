import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 text-sm font-medium whitespace-nowrap transition-all outline-none focus-visible:ring-[3px] focus-visible:ring-[var(--rh-color-accent-base-on-light,#0066cc)]/30 dark:focus-visible:ring-[var(--rh-color-accent-base-on-dark,#92c5f9)]/30 disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
  {
    variants: {
      variant: {
        default:
          "rounded-[var(--rh-border-radius-default,3px)] bg-[var(--rh-color-accent-base-on-light,#0066cc)] text-white hover:bg-[#004d99] dark:bg-[var(--rh-color-accent-base-on-dark,#92c5f9)] dark:text-[#151515] dark:hover:bg-[#b9dafc]",
        destructive:
          "rounded-[var(--rh-border-radius-default,3px)] bg-[var(--rh-color-status-danger-on-light,#b1380b)] text-white hover:bg-[#8f2d09] dark:bg-[var(--rh-color-status-danger-on-dark,#f4784a)] dark:text-[#151515] dark:hover:bg-[#f89b78]",
        outline:
          "rounded-[var(--rh-border-radius-default,3px)] border border-border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground dark:border-input dark:bg-input/30 dark:hover:bg-input/50",
        secondary:
          "rounded-[var(--rh-border-radius-default,3px)] bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost:
          "rounded-[var(--rh-border-radius-default,3px)] hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        link: "text-[var(--rh-color-accent-base-on-light,#0066cc)] dark:text-[var(--rh-color-accent-base-on-dark,#92c5f9)] underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2 has-[>svg]:px-3",
        xs: "h-6 gap-1 px-2 text-xs has-[>svg]:px-1.5 [&_svg:not([class*='size-'])]:size-3",
        sm: "h-8 gap-1.5 px-3 has-[>svg]:px-2.5",
        lg: "h-10 px-6 has-[>svg]:px-4",
        icon: "size-9",
        "icon-xs": "size-6 [&_svg:not([class*='size-'])]:size-3",
        "icon-sm": "size-8",
        "icon-lg": "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
