import { useState } from "react"
import { ChevronDown, Check } from "lucide-react"
import type { ToolResult } from "../types"

const TOOL_LABELS: Record<string, string> = {
  "list jobs": "Listed jobs",
  "get job": "Checked job",
  "cancel job": "Cancelled job",
  "get job logs": "Fetched logs",
  "get job artifacts": "Fetched artifacts",
  "list recipes": "Checked recipes",
  "get recipe": "Loaded recipe",
  "submit recipe job": "Submitted job",
  "get config": "Checked config",
  "estimate sdg cost": "Estimated SDG cost",
  "compare sdg models": "Comparing teacher models",
  "estimate training cost": "Estimating training cost",
  "estimate training method cost": "Estimating training methods",
  "estimate eval cost": "Estimating evaluation cost",
  "get job detail": "Checked job details",
  "get jobs": "Listed jobs",
}

function humanize(name: string): string {
  const key = name.replace(/_/g, " ").toLowerCase()
  return TOOL_LABELS[key] ?? name.replace(/_/g, " ")
}

function dedup(tools: ToolResult[]): ToolResult[] {
  const seen = new Map<string, ToolResult>()
  for (const t of tools) {
    const key = t.name.replace(/_/g, " ").toLowerCase()
    const existing = seen.get(key)
    if (!existing || (!existing.result && t.result)) {
      seen.set(key, t)
    }
  }
  return [...seen.values()]
}

interface ToolActivityProps {
  toolResults: ToolResult[]
}

export function ToolActivity({ toolResults }: ToolActivityProps) {
  const [expanded, setExpanded] = useState(false)
  const unique = dedup(toolResults)
  if (unique.length === 0) return null

  const summary = unique.map((t) => humanize(t.name)).join(" · ")

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        className="group flex items-center gap-1.5 text-[11px] text-muted-foreground/50 transition-colors hover:text-muted-foreground"
      >
        <span className="h-1 w-1 rounded-full bg-current shrink-0" />
        <span>{summary}</span>
        <ChevronDown
          className={`h-3 w-3 opacity-0 group-hover:opacity-100 transition-all duration-200 ${
            expanded ? "rotate-180" : ""
          }`}
        />
      </button>
      {expanded && (
        <div className="mt-1.5 rounded-lg border border-border/30 bg-rh-blue-light/50 dark:bg-rh-blue-dark/20 py-1">
          {unique.map((tool, i) => (
            <ToolDetailRow key={i} tool={tool} />
          ))}
        </div>
      )}
    </div>
  )
}

function ToolDetailRow({ tool }: { tool: ToolResult }) {
  const [showResult, setShowResult] = useState(false)

  return (
    <div className="px-2.5 py-1">
      <button
        type="button"
        onClick={() => tool.result && setShowResult((s) => !s)}
        className={`flex items-center gap-2 text-[11px] w-full text-left ${
          tool.result ? "cursor-pointer hover:text-foreground" : "cursor-default"
        } text-muted-foreground transition-colors`}
      >
        <Check className="h-3 w-3 text-muted-foreground/40 shrink-0" />
        <span className="h-1.5 w-1.5 rounded-full bg-rh-green shrink-0" />
        <span className="font-mono">{tool.name}</span>
        {tool.result && (
          <ChevronDown
            className={`h-2.5 w-2.5 ml-auto opacity-50 transition-transform duration-200 ${
              showResult ? "rotate-180" : ""
            }`}
          />
        )}
      </button>
      {showResult && tool.result && (
        <pre className="mt-1 ml-5 max-h-32 overflow-auto rounded bg-background/80 p-2 text-[10px] text-muted-foreground/70 leading-relaxed">
          {tool.result}
        </pre>
      )}
    </div>
  )
}

// Keep legacy export for backwards compat
export function ToolBadge({ toolResult }: { toolResult: ToolResult }) {
  return <ToolActivity toolResults={[toolResult]} />
}
