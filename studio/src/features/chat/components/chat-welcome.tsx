import { Bot, Cpu, Database, Rocket, BarChart3 } from "lucide-react"

interface ChatWelcomeProps {
  onPromptClick: (prompt: string) => void
}

const CAPABILITIES = [
  {
    icon: Rocket,
    label: "Build custom agents",
    detail: "Classification, extraction, and more",
    prompt: "I want to build a custom agent",
    iconBg: "bg-[#fce3e3] dark:bg-[#731f00]/50",
    iconColor: "text-[#ee0000] dark:text-[#f56e6e]",
    hoverBorder: "hover:border-[#ee0000]/30 dark:hover:border-[#f56e6e]/30",
  },
  {
    icon: Database,
    label: "Generate training data",
    detail: "Synthetic data generation (SDG)",
    prompt: "Help me generate training data",
    iconBg: "bg-[#ece6ff] dark:bg-[#1b0d33]/50",
    iconColor: "text-[#5e40be] dark:text-[#876fd4]",
    hoverBorder: "hover:border-[#5e40be]/30 dark:hover:border-[#876fd4]/30",
  },
  {
    icon: Cpu,
    label: "Train models",
    detail: "Fine-tune on your infrastructure",
    prompt: "I want to train a model",
    iconBg: "bg-[#e0f0ff] dark:bg-[#003366]/50",
    iconColor: "text-[#0066cc] dark:text-[#4394e5]",
    hoverBorder: "hover:border-[#0066cc]/30 dark:hover:border-[#4394e5]/30",
  },
  {
    icon: BarChart3,
    label: "Evaluate performance",
    detail: "Compare against frontier baselines",
    prompt: "Help me evaluate a model's performance",
    iconBg: "bg-[#daf2f2] dark:bg-[#003333]/50",
    iconColor: "text-[#147878] dark:text-[#37a3a3]",
    hoverBorder: "hover:border-[#147878]/30 dark:hover:border-[#37a3a3]/30",
  },
]

const SUGGESTED_PROMPTS = [
  "Help me build a support ticket classifier",
  "Create an agent that extracts invoice data",
  "I want to train a model for sentiment analysis",
]

export function ChatWelcome({ onPromptClick }: ChatWelcomeProps) {
  return (
    <div className="flex h-full flex-col items-center justify-start px-6 pb-12 pt-10">
      <div className="w-full max-w-xl space-y-8">
        <div className="animate-welcome text-center space-y-3">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-rh-red mb-4">
            <Bot className="h-7 w-7 text-white" />
          </div>
          <h2 className="text-2xl font-semibold tracking-tight">
            Welcome to Amortized Studio
          </h2>
          <p className="text-muted-foreground text-base leading-relaxed max-w-sm mx-auto">
            I'm Morty, your AI assistant for building task-specific models.
          </p>
        </div>

        <div
          className="animate-welcome grid grid-cols-2 gap-2"
          style={{ animationDelay: "200ms" }}
        >
          {CAPABILITIES.map((cap, i) => (
            <button
              key={cap.label}
              onClick={() => onPromptClick(cap.prompt)}
              className={`animate-message-in flex items-start gap-2.5 rounded-lg border border-border/40 px-3 py-2.5 text-left transition-all duration-300 hover:shadow-md hover:-translate-y-0.5 ${cap.hoverBorder}`}
              style={{ animationDelay: `${250 + i * 100}ms` }}
            >
              <div className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${cap.iconBg}`}>
                <cap.icon className={`h-3.5 w-3.5 ${cap.iconColor}`} />
              </div>
              <div className="min-w-0">
                <p className="text-xs font-medium leading-tight">{cap.label}</p>
                <p className="text-[11px] text-muted-foreground leading-tight mt-0.5">{cap.detail}</p>
              </div>
            </button>
          ))}
        </div>

        <div className="space-y-3">
          <p
            className="animate-welcome text-xs font-medium text-muted-foreground uppercase tracking-wider text-center"
            style={{ animationDelay: "400ms" }}
          >
            Try asking
          </p>
          <div className="grid gap-2">
            {SUGGESTED_PROMPTS.map((prompt, i) => (
              <button
                key={prompt}
                onClick={() => onPromptClick(prompt)}
                className="animate-message-in group relative overflow-hidden rounded-xl border bg-card px-4 py-3 text-left transition-all duration-300 hover:border-primary/40 hover:bg-accent hover:shadow-md hover:-translate-y-0.5 dark:hover:border-primary/50"
                style={{ animationDelay: `${500 + i * 150}ms` }}
              >
                <span className="text-sm font-medium leading-relaxed">
                  {prompt}
                </span>
              </button>
            ))}
          </div>
        </div>

        <p
          className="animate-welcome text-center text-sm text-muted-foreground"
          style={{ animationDelay: "1000ms" }}
        >
          Or type your own question below
        </p>
      </div>
    </div>
  )
}
