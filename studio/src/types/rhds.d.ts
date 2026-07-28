import type { CSSProperties, HTMLAttributes } from "react"

type RhColorPalette =
  | "lightest"
  | "lighter"
  | "light"
  | "dark"
  | "darker"
  | "darkest"

type RhBaseProps = {
  id?: string
  className?: string
  style?: CSSProperties
  slot?: string
  children?: React.ReactNode
} & HTMLAttributes<HTMLElement>

declare module "react" {
  namespace JSX {
    interface IntrinsicElements {
      "rh-alert": RhBaseProps & {
        state?: "default" | "info" | "success" | "warning" | "danger"
        variant?: "inline" | "toast"
        dismissable?: boolean
      }
      "rh-accordion": RhBaseProps & {
        accent?: "inline" | "bottom"
        large?: boolean
        "color-palette"?: RhColorPalette
        "expanded-index"?: string
      }
      "rh-accordion-header": RhBaseProps & {
        expanded?: boolean
      }
      "rh-accordion-panel": RhBaseProps
      "rh-badge": RhBaseProps & {
        number?: number
        threshold?: number
        state?:
          | "neutral"
          | "info"
          | "success"
          | "warning"
          | "caution"
          | "danger"
      }
      "rh-button": RhBaseProps & {
        variant?: "primary" | "secondary" | "tertiary" | "close" | "play"
        danger?: boolean
        disabled?: boolean
        type?: "button" | "submit" | "reset"
        icon?: string
        "icon-set"?: string
        "accessible-label"?: string
      }
      "rh-card": RhBaseProps & {
        "color-palette"?: RhColorPalette
        variant?: "promo"
        "full-width"?: boolean
      }
      "rh-dialog": RhBaseProps & {
        variant?: "small" | "medium" | "large"
        position?: "top"
        open?: boolean
        trigger?: string
        type?: "video"
        "accessible-label"?: string
        ref?: React.Ref<HTMLElement & { showModal: () => void; close: () => void }>
      }
      "rh-icon": RhBaseProps & {
        set?: string
        icon?: string
        size?: "sm" | "md" | "lg" | "xl"
      }
      "rh-select": RhBaseProps & {
        "accessible-label"?: string
      }
      "rh-skeleton": RhBaseProps
      "rh-spinner": RhBaseProps & {
        size?: "sm" | "md" | "lg" | "xl"
      }
      "rh-surface": RhBaseProps & {
        "color-palette"?: RhColorPalette
      }
      "rh-switch": RhBaseProps & {
        checked?: boolean
        disabled?: boolean
        "accessible-label"?: string
        "show-check-icon"?: boolean
        "message-on"?: string
        "message-off"?: string
      }
      "rh-table": RhBaseProps & {
        "color-palette"?: "lightest" | "light" | "dark" | "darkest"
      }
      "rh-tabs": RhBaseProps & {
        "active-index"?: number
        manual?: boolean
        vertical?: boolean
        box?: "box" | "inset"
        centered?: boolean
        "color-palette"?: RhColorPalette
      }
      "rh-tab": RhBaseProps & {
        active?: boolean
        disabled?: boolean
      }
      "rh-tab-panel": RhBaseProps
      "rh-tag": RhBaseProps & {
        "color-palette"?: RhColorPalette
        variant?: "filled" | "outline" | "desaturated"
      }
      "rh-tooltip": RhBaseProps & {
        position?: "top" | "right" | "bottom" | "left" | "top-start" | "top-end" | "bottom-start" | "bottom-end" | "left-start" | "left-end" | "right-start" | "right-end"
        "accessible-text"?: string
      }
      "rh-scheme-toggle": RhBaseProps
      "rh-sort-button": RhBaseProps
    }
  }
}

export {}
