import { useCallback, useEffect, useRef, useState } from "react"
import { useNavigate, useLocation } from "react-router"
import {
  Bot,
  Check,
  ArrowRight,
  ArrowLeft,
  Server,
  Zap,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { useUIStore } from "@/stores/ui-store"
import { cn } from "@/lib/utils"
import { mlflowUiHref } from "@/lib/api-client"
import { TUTORIAL_STEPS } from "./tutorial-steps"

interface TargetRect {
  top: number
  left: number
  width: number
  height: number
}

const PAD = 8
const GAP = 28
const TOOLTIP_W = 320
const RECIPE_TOOLTIP_W = 260
const MARGIN = 16
const EDGE = 6
const FADE_MS = 450

const SPOTLIGHT_COUNT = TUTORIAL_STEPS.filter((s) => !!s.target).length

function getSpotlightIndex(stepIndex: number): number {
  let idx = 0
  for (let i = 0; i < stepIndex; i++) {
    if (TUTORIAL_STEPS[i]?.target) idx++
  }
  return idx
}

function StepDots({ current, total }: { current: number; total: number }) {
  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: total }, (_, i) => (
        <span
          key={i}
          className={cn(
            "block h-1.5 w-1.5 rounded-full transition-all duration-300",
            i === current && "bg-primary scale-125",
            i < current && "bg-primary/50",
            i > current && "bg-muted-foreground/25",
          )}
        />
      ))}
    </div>
  )
}

function computeTooltipPos(
  st: number,
  sl: number,
  sw: number,
  sh: number,
  placement: "top" | "right" | "bottom" | "left" = "bottom",
  estH = 200,
): { top: number; left: number } {
  const vw = window.innerWidth
  const vh = window.innerHeight

  let top = 0
  let left = 0

  switch (placement) {
    case "top":
      top = st - GAP - estH
      left = sl + sw / 2 - TOOLTIP_W / 2
      break
    case "bottom":
      top = st + sh + GAP
      left = sl
      break
    case "left":
      left = sl - GAP - TOOLTIP_W
      top = st
      if (left < MARGIN) {
        left = sl + sw + GAP
        if (left + TOOLTIP_W > vw - MARGIN) {
          top = st + sh + GAP
          left = sl
        }
      }
      break
    case "right":
      left = sl + sw + GAP
      top = st
      break
  }

  if (left < MARGIN) left = MARGIN
  if (left + TOOLTIP_W > vw - MARGIN) left = vw - MARGIN - TOOLTIP_W
  if (top < MARGIN) top = MARGIN
  if (top + estH > vh - MARGIN) top = vh - MARGIN - estH

  return { top, left }
}

export function TutorialOverlay() {
  const navigate = useNavigate()
  const location = useLocation()
  const active = useUIStore((s) => s.tutorialActive)
  const stepIndex = useUIStore((s) => s.tutorialStep)
  const nextStep = useUIStore((s) => s.nextTutorialStep)
  const prevStep = useUIStore((s) => s.prevTutorialStep)
  const completeTutorial = useUIStore((s) => s.completeTutorial)
  const skipTutorial = useUIStore((s) => s.skipTutorial)

  const totalSteps = TUTORIAL_STEPS.length

  // displayedStepIndex lags behind stepIndex — it only updates after
  // the fade-out completes, so the tooltip stays at its old position
  // and shows its old content during the entire fade-out transition.
  const [displayedStepIndex, setDisplayedStepIndex] = useState(stepIndex)
  const [rect, setRect] = useState<TargetRect | null>(null)
  const [tooltipVisible, setTooltipVisible] = useState(false)
  const timersRef = useRef<number[]>([])
  const tooltipRef = useRef<HTMLDivElement>(null)
  const [tooltipH, setTooltipH] = useState(200)

  const step = TUTORIAL_STEPS[stepIndex]
  const displayedStep = TUTORIAL_STEPS[displayedStepIndex]

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(clearTimeout)
    timersRef.current = []
  }, [])

  const addTimer = useCallback((fn: () => void, ms: number) => {
    timersRef.current.push(window.setTimeout(fn, ms))
  }, [])

  const findTarget = useCallback(
    (targetSelector: string) => {
      let attempts = 0
      const maxAttempts = 60

      const poll = () => {
        const el = document.querySelector<HTMLElement>(
          `[data-tutorial="${targetSelector}"]`,
        )
        if (el) {
          window.scrollTo({ top: 0, behavior: "smooth" })
          addTimer(() => {
            const r = el.getBoundingClientRect()
            setRect({
              top: r.top,
              left: r.left,
              width: r.width,
              height: r.height,
            })
            addTimer(() => setTooltipVisible(true), 400)
          }, 150)
          return
        }
        attempts++
        if (attempts >= maxAttempts) {
          setRect(null)
          return
        }
        requestAnimationFrame(poll)
      }

      requestAnimationFrame(poll)
    },
    [addTimer],
  )

  useEffect(() => {
    if (!active || !step) return
    clearTimers()
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: kick off fade-out before timer swaps content
    setTooltipVisible(false)

    // Wait for fade-out to finish, THEN switch displayed content and position.
    addTimer(() => {
      setDisplayedStepIndex(stepIndex)
      setRect(null)

      if (!step.target) {
        addTimer(() => setTooltipVisible(true), 150)
        return
      }

      if (step.route && location.pathname !== step.route) {
        navigate(step.route)
        addTimer(() => findTarget(step.target!), 250)
      } else {
        findTarget(step.target!)
      }
    }, FADE_MS)

    return clearTimers
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, stepIndex])

  useEffect(() => {
    if (!active || !step?.target || !rect) return

    const handleResize = () => {
      const el = document.querySelector<HTMLElement>(
        `[data-tutorial="${step.target}"]`,
      )
      if (el) {
        const r = el.getBoundingClientRect()
        setRect({
          top: r.top,
          left: r.left,
          width: r.width,
          height: r.height,
        })
      }
    }

    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [active, step, rect])

  useEffect(() => {
    if (!active) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        skipTutorial()
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        if (stepIndex < totalSteps - 1) {
          nextStep()
        }
      } else if (e.key === "ArrowLeft") {
        if (stepIndex > 0) {
          prevStep()
        }
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [active, stepIndex, totalSteps, nextStep, prevStep, skipTutorial])

  useEffect(() => {
    if (!tooltipRef.current) return
    const h = tooltipRef.current.offsetHeight
    if (h > 0) setTooltipH(h)
  }, [tooltipVisible, displayedStepIndex])

  useEffect(() => {
    return clearTimers
  }, [clearTimers])

  if (!active || !displayedStep) return null

  const spotlightIdx = getSpotlightIndex(displayedStepIndex)
  const isFirstSpotlight = spotlightIdx === 0
  const isLastSpotlight = spotlightIdx === SPOTLIGHT_COUNT - 1

  // ── MODAL STEPS ──
  if (!displayedStep.target) {
    const isWelcome = displayedStep.id === "welcome"
    const isComplete = displayedStep.id === "complete"

    return (
      <div
        className="fixed inset-0 z-[9998]"
        style={{ backgroundColor: "rgba(0,0,0,0.55)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          className="fixed bg-card rounded-2xl shadow-2xl p-6 z-[10002]"
          style={{
            top: "50%",
            left: "50%",
            maxWidth: "24rem",
            width: "calc(100% - 2rem)",
            opacity: tooltipVisible ? 1 : 0,
            transform: tooltipVisible
              ? "translate(-50%, -50%) scale(1)"
              : "translate(-50%, -50%) scale(0.96)",
            transition: `opacity ${FADE_MS}ms ease-out, transform ${FADE_MS}ms cubic-bezier(0.16, 1, 0.3, 1)`,
          }}
        >
          {isWelcome && (
            <div className="flex flex-col items-center text-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-rh-red shadow-lg shadow-rh-red/20">
                <Bot className="h-7 w-7 text-white" />
              </div>
              <h2 className="text-xl font-display font-semibold">
                {displayedStep.title}
              </h2>
              <p className="text-sm text-muted-foreground">
                {displayedStep.description}
              </p>
              <Button className="w-full" onClick={nextStep}>
                Start Tour
              </Button>
              <button
                className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                onClick={skipTutorial}
              >
                Skip tour
              </button>
            </div>
          )}

          {isComplete && (
            <div className="flex flex-col items-center text-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#3d7317] dark:bg-[#4a8c1f]">
                <Check className="h-7 w-7 text-white" />
              </div>
              <h2 className="text-xl font-display font-semibold">
                {displayedStep.title}
              </h2>
              <p className="text-sm text-muted-foreground">
                {displayedStep.description}
              </p>
              <div className="flex w-full flex-col gap-2">
                <Button
                  className="w-full"
                  onClick={() => {
                    completeTutorial()
                    navigate("/chat")
                  }}
                >
                  Start Chatting
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => {
                    completeTutorial()
                    navigate("/recipes")
                  }}
                >
                  Browse Recipes
                </Button>
                <Button
                  variant="ghost"
                  className="w-full text-muted-foreground"
                  onClick={() => {
                    completeTutorial()
                    navigate("/overview")
                  }}
                >
                  Back to Overview
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  // ── SPOTLIGHT STEPS ──
  // For the sidebar, shrink the spotlight inward so it stays within the
  // visible sidebar boundary instead of bleeding into the content area.
  const isSidebar = displayedStep.id === "sidebar"
  const pad = isSidebar ? -2 : PAD

  const st = Math.max(EDGE, (rect?.top ?? 0) - pad)
  const sl = Math.max(EDGE, (rect?.left ?? 0) - pad)
  const sw = rect
    ? Math.min(rect.width + pad * 2, window.innerWidth - sl - EDGE)
    : 0
  const sh = rect
    ? Math.min(rect.height + pad * 2, window.innerHeight - st - EDGE)
    : 0

  const isRecipeStep = displayedStep.id.startsWith("recipe-step")
  const isRecipeBrowser = displayedStep.id === "recipes"
  const tooltipWidth = isRecipeStep || isRecipeBrowser ? RECIPE_TOOLTIP_W : TOOLTIP_W

  let tooltipPos: { top: number; left: number }

  if (isRecipeBrowser && rect) {
    tooltipPos = {
      top: Math.max(MARGIN, st + sh / 2 - 110),
      left: MARGIN,
    }
  } else if (isRecipeStep && rect) {
    const stepCenterY = st + sh / 2
    tooltipPos = {
      top: Math.max(MARGIN, stepCenterY - 110),
      left: Math.max(MARGIN, sl - GAP - RECIPE_TOOLTIP_W),
    }
  } else if (rect) {
    tooltipPos = computeTooltipPos(st, sl, sw, sh, displayedStep.placement, tooltipH)
  } else {
    tooltipPos = {
      top: window.innerHeight / 2,
      left: window.innerWidth / 2 - TOOLTIP_W / 2,
    }
  }

  return (
    <>
      {/* Click blocker */}
      <div
        className="fixed inset-0 z-[9997]"
        onClick={(e) => e.stopPropagation()}
      />

      {/* Spotlight with rounded cutout */}
      {rect ? (
        <div
          className="fixed z-[9998] rounded-xl pointer-events-none"
          style={{
            top: st,
            left: sl,
            width: sw,
            height: sh,
            boxShadow: "0 0 0 9999px rgba(0,0,0,0.55)",
            backgroundColor: tooltipVisible ? "transparent" : "rgba(0,0,0,0.55)",
            transition: `background-color ${FADE_MS}ms ease-out`,
          }}
        >
          <div
            className="absolute inset-0 rounded-xl tutorial-ring"
            style={{
              boxShadow:
                "inset 0 0 0 2.5px color-mix(in srgb, var(--primary) 50%, transparent), 0 0 16px 2px color-mix(in srgb, var(--primary) 15%, transparent)",
              opacity: tooltipVisible ? 1 : 0,
              transition: `opacity ${FADE_MS}ms ease-out`,
            }}
          />
        </div>
      ) : (
        <div
          className="fixed inset-0 z-[9998]"
          style={{ backgroundColor: "rgba(0,0,0,0.55)" }}
        />
      )}

      {/* Tooltip */}
      <div
        ref={tooltipRef}
        className="fixed z-[10002] bg-card border border-border/80 rounded-xl shadow-xl p-4"
        style={{
          top: tooltipPos.top,
          left: tooltipPos.left,
          width: tooltipWidth,
          opacity: tooltipVisible ? 1 : 0,
          transform: tooltipVisible ? "translateY(0)" : "translateY(10px)",
          transition: `opacity ${FADE_MS}ms ease-out, transform ${FADE_MS}ms cubic-bezier(0.16, 1, 0.3, 1)`,
          pointerEvents: tooltipVisible ? "auto" : "none",
        }}
      >
        <div className="flex flex-col gap-2.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-display uppercase tracking-wider text-muted-foreground/70">
              Step {spotlightIdx + 1} of {SPOTLIGHT_COUNT}
            </span>
            <StepDots current={displayedStepIndex} total={totalSteps} />
          </div>
          <h3 className="text-sm font-display font-semibold leading-snug">
            {displayedStep.title}
          </h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {displayedStep.description}
          </p>

          {displayedStep.id === "settings" && (
            <div className="flex flex-col gap-1.5 pt-1">
              <a
                href="https://github.com/amortized-ai/amortized#readme"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 rounded-lg border border-border/50 px-2.5 py-1.5 text-xs transition-colors hover:bg-muted hover:border-border"
              >
                <Server className="h-3 w-3 text-[#0066cc] dark:text-[#4394e5]" /> Setup Backend &
                MLflow
              </a>
              <a
                href={mlflowUiHref("/gateway")}
                className="flex items-center gap-2 rounded-lg border border-border/50 px-2.5 py-1.5 text-xs transition-colors hover:bg-muted hover:border-border"
              >
                <Zap className="h-3 w-3 text-[#5e40be] dark:text-[#876fd4]" /> Configure AI Gateway
              </a>
            </div>
          )}

          <div className="mt-1 flex items-center gap-2">
            {!isFirstSpotlight && (
              <Button variant="ghost" size="sm" onClick={prevStep}>
                <ArrowLeft className="mr-1 h-3.5 w-3.5" />
                Back
              </Button>
            )}
            <button
              className="text-xs text-muted-foreground hover:text-foreground transition-colors px-2"
              onClick={skipTutorial}
            >
              Skip
            </button>
            <div className="flex-1" />
            <Button
              size="sm"
              onClick={() => {
                if (stepIndex < totalSteps - 1) {
                  nextStep()
                } else {
                  completeTutorial()
                }
              }}
            >
              {isLastSpotlight ? "Finish" : "Next"}
              {!isLastSpotlight && (
                <ArrowRight className="ml-1 h-3.5 w-3.5" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </>
  )
}
