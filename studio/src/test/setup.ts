import "@testing-library/jest-dom/vitest"

if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof Element !== "undefined" && !Element.prototype.scrollIntoView) {
  // jsdom doesn't implement scrollIntoView; cmdk (Command) calls it on mount.
  Element.prototype.scrollIntoView = () => {}
}

if (typeof globalThis.IntersectionObserver === "undefined") {
  globalThis.IntersectionObserver = class IntersectionObserver {
    readonly root = null
    readonly rootMargin = ""
    readonly thresholds: readonly number[] = []
    observe() {}
    unobserve() {}
    disconnect() {}
    takeRecords(): IntersectionObserverEntry[] {
      return []
    }
  } as unknown as typeof IntersectionObserver
}
