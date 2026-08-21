interface ThinkingStepsProps {
  step: string | null
}

export function ThinkingSteps({ step }: ThinkingStepsProps) {
  if (!step) return null

  return (
    <p key={step} className="text-xs text-muted-foreground/70 mt-1.5 animate-message-in">
      {step}
    </p>
  )
}
