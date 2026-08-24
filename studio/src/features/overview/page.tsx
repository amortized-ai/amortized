import { useState } from "react"
import { SectionCards } from "./components/section-cards"
import { GpuAllocationCard } from "./components/gpu-allocation-card"
import { RecentJobs } from "./components/recent-jobs"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { MessageSquare, BookOpen, Briefcase, Database, ArrowRight, ChevronDown, Bot, Server, Layers, Zap, Settings, Sparkles, Compass } from "lucide-react"
import { Link } from "react-router"
import { useDatasets } from "@/features/datasets/api/use-datasets"
import { useModels } from "@/features/models/api/use-models"
import { useJobs } from "@/features/jobs/api/use-jobs"
import { useUIStore } from "@/stores/ui-store"

export default function OverviewPage() {
  const { data: datasets, isLoading: datasetsLoading } = useDatasets()
  const { data: models, isLoading: modelsLoading } = useModels()
  const { data: jobs, isLoading: jobsLoading } = useJobs()
  const [activeStep, setActiveStep] = useState<number | null>(null)

  const isLoading = datasetsLoading || modelsLoading || jobsLoading

  const hasData =
    !isLoading &&
    ((datasets?.length ?? 0) > 0 ||
      (models?.length ?? 0) > 0 ||
      (jobs?.length ?? 0) > 0)

  function toggleStep(step: number) {
    setActiveStep((s) => (s === step ? null : step))
  }

  return (
    <div className="@container/main flex flex-1 flex-col gap-2">
      <div className="flex flex-col gap-14 py-6 md:py-8">

        {/* Hero — centered */}
        <div className="px-4 lg:px-6">
          <div className="animate-welcome mx-auto max-w-2xl text-center space-y-4">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-rh-red shadow-lg shadow-rh-red/20">
              <Bot className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Amortized Studio</h1>
              <p className="text-sm text-muted-foreground mt-2 leading-relaxed max-w-md mx-auto">
                Build task-specific AI models that outperform frontier APIs — at a fraction of the cost. Describe what you need, and Morty walks you through data generation and training.
              </p>
            </div>
            <div className="flex items-center justify-center gap-6 pt-1 text-xs text-muted-foreground/70">
              <span className="inline-flex items-center gap-1.5"><Sparkles className="h-3 w-3 text-[#5e40be] dark:text-[#876fd4]" /> Synthetic data</span>
              <span className="inline-flex items-center gap-1.5"><BookOpen className="h-3 w-3 text-[#0066cc] dark:text-[#4394e5]" /> Fine-tuning</span>
              <span className="inline-flex items-center gap-1.5"><Briefcase className="h-3 w-3 text-[#147878] dark:text-[#37a3a3]" /> On-prem deployment</span>
            </div>
            <div className="flex items-center justify-center gap-2.5 pt-4">
              <button
                onClick={() => useUIStore.getState().startTutorial()}
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#0066cc]/25 bg-[#e0f0ff]/40 px-3.5 py-2 text-xs font-medium text-[#0066cc] transition-all duration-200 hover:bg-[#e0f0ff] hover:border-[#0066cc]/40 hover:shadow-sm dark:border-[#4394e5]/25 dark:bg-[#003366]/30 dark:text-[#4394e5] dark:hover:bg-[#003366]/50"
              >
                <Compass className="h-3.5 w-3.5" />
                Take a Tour
              </button>
              <Link
                to="/chat"
                className="inline-flex items-center gap-1.5 rounded-lg border border-[#ee0000]/20 bg-[#fce3e3]/40 px-3.5 py-2 text-xs font-medium text-[#ee0000] transition-all duration-200 hover:bg-[#fce3e3] hover:border-[#ee0000]/35 hover:shadow-sm dark:border-[#f56e6e]/20 dark:bg-[#731f00]/25 dark:text-[#f56e6e] dark:hover:bg-[#731f00]/40"
              >
                <MessageSquare className="h-3.5 w-3.5" />
                Chat with Morty
              </Link>
              <Link
                to="/recipes"
                className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 bg-muted/30 px-3.5 py-2 text-xs font-medium text-muted-foreground transition-all duration-200 hover:bg-muted/60 hover:border-border hover:text-foreground hover:shadow-sm"
              >
                <BookOpen className="h-3.5 w-3.5" />
                Browse Recipes
              </Link>
            </div>
          </div>
        </div>

        {/* Setup section */}
        <div className="px-4 lg:px-6 space-y-6">
          <div className="flex items-center gap-4">
            <div className="h-px flex-1 bg-border/50" />
            <h2 className="text-sm font-semibold text-muted-foreground/70 uppercase tracking-wider shrink-0">Setup</h2>
            <div className="h-px flex-1 bg-border/50" />
          </div>
          <div
            className="animate-message-in rounded-xl border bg-card p-5 space-y-4"
            style={{ animationDelay: "300ms" }}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Before you start</p>
                <h3 className="text-sm font-semibold">Connect your services</h3>
              </div>
              <Button asChild variant="outline" size="sm">
                <Link to="/settings" className="gap-1.5">
                  <Settings className="h-3.5 w-3.5" />
                  Check Settings
                </Link>
              </Button>
            </div>

            <div className="grid gap-3 md:grid-cols-3">
              <Link to="/settings" className="group flex items-start gap-2.5 rounded-lg border border-border/50 bg-muted/20 px-3 py-2.5 transition-all duration-300 hover:border-[#b9dafc] hover:bg-[#e0f0ff]/30 dark:hover:border-[#003366] dark:hover:bg-[#003366]/20 cursor-pointer">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#e0f0ff] text-[#0066cc] transition-colors duration-300 group-hover:bg-[#b9dafc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                  <Server className="h-3.5 w-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">Backend API</p>
                  <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">Runs jobs and manages recipes. Start with <code className="bg-muted px-1 py-0.5 rounded text-xs font-mono">amortized serve</code></p>
                </div>
                <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/0 group-hover:text-muted-foreground transition-all duration-300 mt-1" />
              </Link>
              <Link to="/settings" className="group flex items-start gap-2.5 rounded-lg border border-border/50 bg-muted/20 px-3 py-2.5 transition-all duration-300 hover:border-[#d0c5f4] hover:bg-[#ece6ff]/30 dark:hover:border-[#21134d] dark:hover:bg-[#1b0d33]/20 cursor-pointer">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#ece6ff] text-[#5e40be] transition-colors duration-300 group-hover:bg-[#d0c5f4] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
                  <Layers className="h-3.5 w-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">MLflow</p>
                  <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">Tracks experiments, stores datasets and models</p>
                </div>
                <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/0 group-hover:text-muted-foreground transition-all duration-300 mt-1" />
              </Link>
              <Link to="/settings" className="group flex items-start gap-2.5 rounded-lg border border-border/50 bg-muted/20 px-3 py-2.5 transition-all duration-300 hover:border-[#ffe072] hover:bg-[#fff4cc]/30 dark:hover:border-[#73480b] dark:hover:bg-[#54330b]/20 cursor-pointer">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[#fff4cc] text-[#b98412] transition-colors duration-300 group-hover:bg-[#ffe072] dark:bg-[#54330b]/40 dark:text-[#dca614]">
                  <Zap className="h-3.5 w-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium">AI Gateway</p>
                  <p className="text-xs text-muted-foreground mt-0.5 leading-relaxed">Routes to local models or cloud LLMs for data generation</p>
                </div>
                <ArrowRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground/0 group-hover:text-muted-foreground transition-all duration-300 mt-1" />
              </Link>
            </div>

            <div className="flex items-center justify-between border-t pt-3">
              <p className="text-xs text-muted-foreground">
                <span className="font-semibold text-foreground/80">All three are required</span> for the full workflow.
              </p>
              <Button asChild variant="outline" size="sm" className="h-7 text-xs gap-1.5">
                <a
                  href="https://github.com/amortized-ai/amortized#readme"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Full setup guide <ArrowRight className="h-3 w-3" />
                </a>
              </Button>
            </div>
          </div>
        </div>

        {/* Pipeline section */}
        <div className="px-4 lg:px-6 space-y-6">
          <div className="flex items-center gap-4">
            <div className="h-px flex-1 bg-border/50" />
            <h2 className="text-sm font-semibold text-muted-foreground/70 uppercase tracking-wider shrink-0">Your workflow</h2>
            <div className="h-px flex-1 bg-border/50" />
          </div>
          <div className="space-y-0">

            {/* ── Step 1: Start Building ── */}
            <div className="animate-message-in relative pb-6" style={{ animationDelay: "500ms" }}>
              <button
                onClick={() => toggleStep(1)}
                className="flex items-center gap-3 mb-3 group/step cursor-pointer"
              >
                <div className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ring-4 ring-background z-10 transition-all duration-300",
                  activeStep === 1
                    ? "bg-[#0066cc] dark:bg-[#4394e5] text-white scale-110 shadow-md shadow-[#0066cc]/30 dark:shadow-[#4394e5]/30"
                    : activeStep !== null
                      ? "bg-foreground/60 text-background group-hover/step:bg-foreground/80 group-hover/step:scale-105"
                      : "bg-[#0066cc] dark:bg-[#4394e5] text-white group-hover/step:scale-105 group-hover/step:shadow-md group-hover/step:shadow-[#0066cc]/20 dark:group-hover/step:shadow-[#4394e5]/20",
                )}>
                  1
                </div>
                <p className={cn(
                  "text-xs font-medium uppercase tracking-wider transition-colors duration-200",
                  activeStep === 1
                    ? "text-[#0066cc] dark:text-[#4394e5]"
                    : activeStep !== null
                      ? "text-muted-foreground/50"
                      : "text-[#0066cc]/70 dark:text-[#4394e5]/70 group-hover/step:text-[#0066cc] dark:group-hover/step:text-[#4394e5]",
                )}>
                  Start building
                </p>
              </button>
              <div className={cn(
                "ml-11 rounded-xl p-4 -m-4 transition-all duration-300",
                activeStep === 1
                  ? "border-2 border-[#b9dafc] bg-[#e0f0ff]/20 shadow-sm dark:border-[#003366]/60 dark:bg-[#003366]/10"
                  : "border-2 border-transparent",
                              )}>
                <div className="grid gap-3 md:grid-cols-2">
                  <Link
                    to="/chat"
                    className="group rounded-xl border bg-card p-5 transition-all duration-300 hover:border-[#92c5f9] hover:shadow-md hover:-translate-y-0.5 dark:hover:border-[#003366]"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#e0f0ff] text-[#0066cc] transition-colors duration-300 group-hover:bg-[#b9dafc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                        <MessageSquare className="h-4 w-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-sm">Chat with Morty</h3>
                        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                          Describe your task in plain language — Morty handles data generation and training.
                        </p>
                        <span className="inline-flex items-center gap-1 text-xs font-medium text-[#0066cc] dark:text-[#4394e5] mt-2 transition-all duration-300 group-hover:gap-1.5">
                          Open Chat <ArrowRight className="h-3 w-3" />
                        </span>
                      </div>
                    </div>
                  </Link>

                  <Link
                    to="/recipes"
                    className="group rounded-xl border bg-card p-5 transition-all duration-300 hover:border-[#92c5f9] hover:shadow-md hover:-translate-y-0.5 dark:hover:border-[#003366]"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#e0f0ff] text-[#0066cc] transition-colors duration-300 group-hover:bg-[#b9dafc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
                        <BookOpen className="h-4 w-4" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h3 className="font-semibold text-sm">Use a Recipe</h3>
                        <p className="text-sm text-muted-foreground mt-1 leading-relaxed">
                          Configure training or SDG jobs directly with full control over parameters.
                        </p>
                        <span className="inline-flex items-center gap-1 text-xs font-medium text-[#0066cc] dark:text-[#4394e5] mt-2 transition-all duration-300 group-hover:gap-1.5">
                          Browse Recipes <ArrowRight className="h-3 w-3" />
                        </span>
                      </div>
                    </div>
                  </Link>
                </div>
              </div>
            </div>

            {/* Down arrow → advances to step 2 */}
            <div className="pb-6">
              <button
                onClick={() => setActiveStep(2)}
                className="flex items-center gap-3 group/arrow cursor-pointer"
                aria-label="Go to step 2: Monitor jobs"
              >
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground ring-4 ring-background transition-all duration-300 group-hover/arrow:bg-[#e0f0ff] group-hover/arrow:text-[#0066cc] group-hover/arrow:scale-110 dark:group-hover/arrow:bg-[#003366]/40 dark:group-hover/arrow:text-[#4394e5]">
                  <ChevronDown className="h-3.5 w-3.5" />
                </div>
                <p className="text-xs text-muted-foreground/50 uppercase tracking-widest group-hover/arrow:text-[#0066cc]/60 dark:group-hover/arrow:text-[#4394e5]/60 transition-colors duration-200">then</p>
              </button>
            </div>

            {/* ── Step 2: Monitor ── */}
            <div className="animate-message-in relative pb-6" style={{ animationDelay: "700ms" }}>
              <button
                onClick={() => toggleStep(2)}
                className="flex items-center gap-3 mb-3 group/step cursor-pointer"
              >
                <div className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ring-4 ring-background z-10 transition-all duration-300",
                  activeStep === 2
                    ? "bg-[#0066cc] dark:bg-[#4394e5] text-white scale-110 shadow-md shadow-[#0066cc]/30 dark:shadow-[#4394e5]/30"
                    : activeStep !== null
                      ? "bg-foreground/60 text-background group-hover/step:bg-foreground/80 group-hover/step:scale-105"
                      : "bg-foreground text-background group-hover/step:scale-105 group-hover/step:shadow-md",
                )}>
                  2
                </div>
                <p className={cn(
                  "text-xs font-medium uppercase tracking-wider transition-colors duration-200",
                  activeStep === 2
                    ? "text-[#0066cc] dark:text-[#4394e5]"
                    : activeStep !== null
                      ? "text-muted-foreground/50"
                      : "text-muted-foreground group-hover/step:text-foreground",
                )}>
                  Monitor jobs
                </p>
              </button>
              <div className={cn(
                "ml-11 rounded-xl p-4 -m-4 transition-all duration-300",
                activeStep === 2
                  ? "border-2 border-[#b9dafc] bg-[#e0f0ff]/20 shadow-sm dark:border-[#003366]/60 dark:bg-[#003366]/10"
                  : "border-2 border-transparent",
                              )}>
                <Link
                  to="/jobs"
                  className="group block rounded-xl border bg-card p-5 transition-all duration-300 hover:border-border/80 hover:shadow-md hover:-translate-y-0.5"
                >
                  <div className="flex items-start gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground transition-colors duration-300 group-hover:bg-accent group-hover:text-foreground">
                      <Briefcase className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Track SDG and training jobs in real-time. View logs, metrics, and status updates as they run.
                      </p>
                      <span className="inline-flex items-center gap-1 text-xs font-medium mt-2 transition-all duration-300 group-hover:gap-1.5">
                        View Jobs <ArrowRight className="h-3 w-3" />
                      </span>
                    </div>
                  </div>
                </Link>
              </div>
            </div>

            {/* Down arrow → advances to step 3 */}
            <div className="pb-6">
              <button
                onClick={() => setActiveStep(3)}
                className="flex items-center gap-3 group/arrow cursor-pointer"
                aria-label="Go to step 3: Manage data and models"
              >
                <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground ring-4 ring-background transition-all duration-300 group-hover/arrow:bg-[#e0f0ff] group-hover/arrow:text-[#0066cc] group-hover/arrow:scale-110 dark:group-hover/arrow:bg-[#003366]/40 dark:group-hover/arrow:text-[#4394e5]">
                  <ChevronDown className="h-3.5 w-3.5" />
                </div>
                <p className="text-xs text-muted-foreground/50 uppercase tracking-widest group-hover/arrow:text-[#0066cc]/60 dark:group-hover/arrow:text-[#4394e5]/60 transition-colors duration-200">then</p>
              </button>
            </div>

            {/* ── Step 3: Manage ── */}
            <div className="animate-message-in" style={{ animationDelay: "900ms" }}>
              <button
                onClick={() => toggleStep(3)}
                className="flex items-center gap-3 mb-3 group/step cursor-pointer"
              >
                <div className={cn(
                  "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-xs font-bold ring-4 ring-background z-10 transition-all duration-300",
                  activeStep === 3
                    ? "bg-[#0066cc] dark:bg-[#4394e5] text-white scale-110 shadow-md shadow-[#0066cc]/30 dark:shadow-[#4394e5]/30"
                    : activeStep !== null
                      ? "bg-foreground/60 text-background group-hover/step:bg-foreground/80 group-hover/step:scale-105"
                      : "bg-foreground text-background group-hover/step:scale-105 group-hover/step:shadow-md",
                )}>
                  3
                </div>
                <p className={cn(
                  "text-xs font-medium uppercase tracking-wider transition-colors duration-200",
                  activeStep === 3
                    ? "text-[#0066cc] dark:text-[#4394e5]"
                    : activeStep !== null
                      ? "text-muted-foreground/50"
                      : "text-muted-foreground group-hover/step:text-foreground",
                )}>
                  Manage data & models
                </p>
              </button>
              <div className={cn(
                "ml-11 rounded-xl p-4 -m-4 transition-all duration-300",
                activeStep === 3
                  ? "border-2 border-[#b9dafc] bg-[#e0f0ff]/20 shadow-sm dark:border-[#003366]/60 dark:bg-[#003366]/10"
                  : "border-2 border-transparent",
                              )}>
                <div className="group rounded-xl border bg-card p-5 transition-all duration-300 hover:border-border/80 hover:shadow-md hover:-translate-y-0.5">
                  <div className="flex items-start gap-3">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-muted text-muted-foreground transition-colors duration-300 group-hover:bg-accent group-hover:text-foreground">
                      <Database className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Browse generated datasets, inspect training data quality, and manage your trained models.
                      </p>
                      <div className="flex items-center gap-4 mt-2">
                        <Button asChild variant="link" size="sm" className="h-auto p-0 gap-1 text-xs">
                          <Link to="/datasets">Datasets <ArrowRight className="h-3 w-3" /></Link>
                        </Button>
                        <Button asChild variant="link" size="sm" className="h-auto p-0 gap-1 text-xs">
                          <Link to="/models">Models <ArrowRight className="h-3 w-3" /></Link>
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {hasData && (
          <>
            <SectionCards />
            <div className="px-4 lg:px-6">
              <GpuAllocationCard />
            </div>
            <div className="px-4 lg:px-6">
              <RecentJobs />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
