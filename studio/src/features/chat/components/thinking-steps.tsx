import { useEffect, useRef, useState } from "react"

interface ThinkingStepsProps {
  steps: string[]
}

export function ThinkingSteps({ steps }: ThinkingStepsProps) {
  const [visibleStep, setVisibleStep] = useState<string | null>(null)
  const [fading, setFading] = useState(false)
  const prevStepRef = useRef<string | null>(null)

  useEffect(() => {
    if (steps.length === 0) return
    const latest = steps[steps.length - 1]!
    if (latest === prevStepRef.current) return
    prevStepRef.current = latest

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
