import { useState } from "react"
import { cn } from "@/lib/utils"
import {
  ChevronDown,
  CircleCheck,
  Circle,
  ClipboardCheck,
  Loader2,
  Database,
  Cpu,
} from "lucide-react"
import type { PhasePlan } from "../types"

const PHASE_STYLES = {
  sdg: {
    icon: Database,
    gradient: "from-rh-purple-light/50 to-rh-purple-light/20 dark:from-rh-purple-dark/20 dark:to-rh-purple-dark/5",
    dot: "bg-rh-purple dark:bg-rh-purple",
    connector: "bg-rh-purple/60",
    badge: "bg-rh-purple-light text-rh-purple-dark dark:bg-rh-purple-dark/40 dark:text-rh-purple",
    iconColor: "text-rh-purple",
    check: "text-rh-purple",
  },
  training: {
    icon: Cpu,
    gradient: "from-rh-blue-light/50 to-rh-blue-light/20 dark:from-rh-blue-dark/20 dark:to-rh-blue-dark/5",
    dot: "bg-rh-blue dark:bg-rh-blue",
    connector: "bg-rh-blue/60",
    badge: "bg-rh-blue-light text-rh-blue-dark dark:bg-rh-blue-dark/40 dark:text-rh-blue",
    iconColor: "text-rh-blue",
    check: "text-rh-blue",
  },
  eval: {
    icon: ClipboardCheck,
    gradient: "from-emerald-100/50 to-emerald-100/20 dark:from-emerald-900/20 dark:to-emerald-900/5",
    dot: "bg-emerald-600 dark:bg-emerald-500",
    connector: "bg-emerald-600/60",
    badge: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-400",
    iconColor: "text-emerald-600 dark:text-emerald-500",
    check: "text-emerald-600 dark:text-emerald-500",
  },
}

interface PlanProgressProps {
  plan: PhasePlan | null
}

export function PlanProgress({ plan }: PlanProgressProps) {
  const [expanded, setExpanded] = useState(false)

  if (!plan) return null

  const styles = PHASE_STYLES[plan.phase]
  const PhaseIcon = styles.icon
  const activeIndex = plan.steps.findIndex((s) => s.status === "active")
  const activeLabel =
    activeIndex >= 0
      ? plan.steps[activeIndex]!.label
      : plan.steps[plan.steps.length - 1]!.label

  return (
    <div className={cn("border-b bg-gradient-to-r", styles.gradient)}>
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-muted/30"
      >
        <PhaseIcon className={cn("h-4 w-4 shrink-0", styles.iconColor)} />
        <span
          className={cn(
            "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
            styles.badge,
          )}
        >
          {plan.label}
        </span>

        <div className="flex items-center gap-0">
          {plan.steps.map((step, i) => (
            <div key={i} className="flex items-center">
              <div
                className={cn(
                  "h-2 w-2 rounded-full",
                  step.status === "completed"
                    ? styles.dot
                    : step.status === "active"
                      ? cn(styles.dot, "animate-pulse")
                      : "bg-muted-foreground/20",
                )}
              />
              {i < plan.steps.length - 1 && (
                <div
                  className={cn(
                    "h-0.5 w-3",
                    plan.steps[i + 1]!.status !== "pending"
                      ? styles.connector
                      : "bg-muted-foreground/15",
                  )}
                />
              )}
            </div>
          ))}
        </div>

        <span className="flex-1 truncate text-xs text-muted-foreground">
          {activeLabel}
        </span>

        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-muted-foreground/60 transition-transform duration-200",
            expanded && "rotate-180",
          )}
        />
      </button>

      {expanded && (
        <div className="space-y-0.5 px-4 pb-3">
          {plan.steps.map((step, i) => (
            <div
              key={i}
              className="flex items-center gap-2.5 py-1"
              title={step.label}
            >
              {step.status === "completed" ? (
                <CircleCheck className={cn("h-4 w-4 shrink-0", styles.check)} />
              ) : step.status === "active" ? (
                <Loader2
                  className={cn(
                    "h-4 w-4 shrink-0 animate-spin",
                    styles.iconColor,
                  )}
                />
              ) : (
                <Circle className="h-4 w-4 shrink-0 text-muted-foreground/25" />
              )}
              <span
                className={cn(
                  "text-sm",
                  step.status === "completed" &&
                    "text-muted-foreground/60 line-through",
                  step.status === "active" && "font-medium text-foreground",
                  step.status === "pending" && "text-muted-foreground",
                )}
              >
                {step.label}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
