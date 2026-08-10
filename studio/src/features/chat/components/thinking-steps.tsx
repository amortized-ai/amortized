import { useEffect, useRef, useState } from "react"

interface ThinkingStepsProps {
  steps: string[]
}

export function ThinkingSteps({ steps }: ThinkingStepsProps) {
  const [visibleStep, setVisibleStep] = useState<string | null>(null)
  const [fading, setFading] = useState(false)
  const prevLenRef = useRef(0)

  useEffect(() => {
    if (steps.length === 0) return
    const latest = steps[steps.length - 1]!
    if (steps.length === prevLenRef.current) return
    prevLenRef.current = steps.length

    if (!visibleStep) {
      setVisibleStep(latest)
      return
    }

    setFading(true)
    const timer = setTimeout(() => {
      setVisibleStep(latest)
      setFading(false)
    }, 300)
    return () => clearTimeout(timer)
  }, [steps, visibleStep])

  if (!visibleStep) return null

  return (
    <p
      className={`text-xs text-muted-foreground/70 mt-1.5 transition-opacity duration-300 ${
        fading ? "opacity-0" : "opacity-100"
      }`}
    >
      {visibleStep}
    </p>
  )
}
