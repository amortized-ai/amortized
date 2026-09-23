import { useCallback, useEffect, useRef } from "react"

interface DragResizeOptions {
  /** Read the current width at drag start (read from the store, not a stale render value). */
  getWidth: () => number
  setWidth: (width: number) => void
  min: number
  max: number
  /**
   * Which way the handle grows the panel. "grow-right" for a handle on the right
   * edge of a left-docked panel (drag right = wider); "grow-left" for a handle on
   * the left edge of a right-docked panel (drag left = wider).
   */
  direction?: "grow-right" | "grow-left"
}

/**
 * Column drag-resize handler. Returns an `onMouseDown` to spread onto a resize
 * handle. Mirrors the chat pop-out panel's resize (pixel widths persisted in a
 * store). Adds an `is-resizing` class to <body> during the drag so width
 * transitions can be suppressed while dragging.
 */
export function useDragResize({
  getWidth,
  setWidth,
  min,
  max,
  direction = "grow-right",
}: DragResizeOptions): (e: React.MouseEvent) => void {
  const isDragging = useRef(false)
  const cleanup = useRef<(() => void) | null>(null)

  useEffect(() => {
    return () => cleanup.current?.()
  }, [])

  return useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault()
      isDragging.current = true
      const startX = e.clientX
      const startWidth = getWidth()

      const onMouseMove = (ev: MouseEvent) => {
        if (!isDragging.current) return
        const delta = direction === "grow-left" ? startX - ev.clientX : ev.clientX - startX
        setWidth(Math.min(max, Math.max(min, startWidth + delta)))
      }

      const onMouseUp = () => {
        isDragging.current = false
        document.removeEventListener("mousemove", onMouseMove)
        document.removeEventListener("mouseup", onMouseUp)
        document.body.style.cursor = ""
        document.body.style.userSelect = ""
        document.body.classList.remove("is-resizing")
        cleanup.current = null
      }

      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"
      document.body.classList.add("is-resizing")
      document.addEventListener("mousemove", onMouseMove)
      document.addEventListener("mouseup", onMouseUp)
      cleanup.current = onMouseUp
    },
    [getWidth, setWidth, min, max, direction],
  )
}
