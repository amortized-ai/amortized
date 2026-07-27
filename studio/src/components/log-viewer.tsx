import { useEffect, useRef, useState, useMemo, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { ArrowDown, Search, X } from "lucide-react"

interface LogViewerProps {
  lines: string[]
  isStreaming?: boolean
  className?: string
}

type LogLevel = "info" | "warn" | "error" | "debug" | "trace" | "separator" | "plain"

// eslint-disable-next-line no-control-regex -- intentionally matching ANSI escape sequences
const ANSI_RE = /\x1b\[[0-9;]*m/g

function stripAnsi(str: string): string {
  return str.replace(ANSI_RE, "")
}

interface ParsedLine {
  level: LogLevel
  prefix: string
  module: string
  body: string
}

function classifyLine(raw: string): ParsedLine {
  const line = stripAnsi(raw)

  if (/^-{3,}\s*stderr\s*-{3,}$/i.test(line.trim())) {
    return { level: "separator", prefix: "", module: "", body: line }
  }

  const levelMatch = line.match(
    /^(INFO|WARNING|WARN|ERROR|Error|DEBUG|CRITICAL|FATAL)\s*[:]\s*/i,
  )
  if (levelMatch) {
    const tag = levelMatch[1]!
    let rest = line.slice(levelMatch[0]!.length)
    const normalized = tag.toUpperCase()

    let mod = ""
    const moduleMatch = rest.match(/^([\w.]+):\s*/)
    if (moduleMatch && moduleMatch[1]!.includes(".")) {
      mod = moduleMatch[1]!
      rest = rest.slice(moduleMatch[0]!.length)
    }

    let level: LogLevel = "info"
    if (normalized === "ERROR" || normalized === "CRITICAL" || normalized === "FATAL") level = "error"
    else if (normalized === "WARNING" || normalized === "WARN") level = "warn"
    else if (normalized === "DEBUG") level = "debug"

    return { level, prefix: tag, module: mod, body: rest }
  }

  if (
    /^\s+(File\s+"|\^|~|\.{3}<)/.test(line) ||
    /^Traceback\s+\(/.test(line) ||
    /^\s+raise\s+/.test(line) ||
    /^\w+(\.\w+)*Error:/.test(line) ||
    /^\w+(\.\w+)*Exception:/.test(line)
  ) {
    return { level: "trace", prefix: "", module: "", body: line }
  }

  return { level: "plain", prefix: "", module: "", body: line }
}

interface ClassifiedLine extends ParsedLine {
  raw: string
  index: number
  isFirstInGroup: boolean
  isLastInGroup: boolean
}

function addGroupInfo(items: Omit<ClassifiedLine, "isFirstInGroup" | "isLastInGroup">[]): ClassifiedLine[] {
  return items.map((item, i) => {
    const isErrorish = item.level === "error" || item.level === "trace"
    const prevErrorish = i > 0 && (items[i - 1]?.level === "error" || items[i - 1]?.level === "trace")
    const nextErrorish = i < items.length - 1 && (items[i + 1]?.level === "error" || items[i + 1]?.level === "trace")
    return {
      ...item,
      isFirstInGroup: isErrorish && !prevErrorish,
      isLastInGroup: isErrorish && !nextErrorish,
    }
  })
}

const LEVEL_COLORS: Record<LogLevel, { badge: string; text: string; border: string }> = {
  info:      { badge: "text-[#6cb6ff] bg-[#6cb6ff]/8",   text: "text-zinc-300",       border: "" },
  warn:      { badge: "text-[#e0af68] bg-[#e0af68]/8",   text: "text-zinc-300",       border: "border-l-[#e0af68]/30" },
  error:     { badge: "text-[#f7768e] bg-[#f7768e]/10",  text: "text-[#ffc0cb]/80",   border: "border-l-[#f7768e]/40" },
  debug:     { badge: "text-zinc-600 bg-zinc-600/8",      text: "text-zinc-600",       border: "" },
  trace:     { badge: "",                                  text: "text-zinc-500",       border: "border-l-[#f7768e]/20" },
  separator: { badge: "",                                  text: "",                     border: "" },
  plain:     { badge: "",                                  text: "text-zinc-400",       border: "" },
}

function LogLine({ item }: { item: ClassifiedLine }) {
  const { level, prefix, module, body, index, isFirstInGroup, isLastInGroup } = item
  const colors = LEVEL_COLORS[level]
  const lineNumber = index + 1

  if (level === "separator") {
    return (
      <div className="flex items-center gap-3 py-3 my-1 select-none" aria-hidden>
        <span className="w-9 shrink-0" />
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-zinc-700/60 to-transparent" />
        <span className="text-[10px] font-mono font-medium uppercase tracking-[0.2em] text-zinc-600">stderr</span>
        <div className="h-px flex-1 bg-gradient-to-r from-transparent via-zinc-700/60 to-transparent" />
      </div>
    )
  }

  const isErrorish = level === "error" || level === "trace"
  const groupClasses = isErrorish
    ? `border-l-2 ${colors.border} ${isFirstInGroup ? "mt-1 pt-1 rounded-tl" : ""} ${isLastInGroup ? "mb-1 pb-1 rounded-bl" : ""} ${level === "error" ? "bg-[#f7768e]/[0.03]" : ""}`
    : "border-l-2 border-transparent"

  return (
    <div className={`group flex items-start hover:bg-white/[0.02] transition-colors duration-75 ${groupClasses}`}>
      <span className="text-[10px] font-mono tabular-nums text-zinc-700 w-9 text-right shrink-0 select-none leading-[22px] group-hover:text-zinc-500 transition-colors duration-75 pr-2">
        {lineNumber}
      </span>
      <div className="flex-1 min-w-0 pl-2 leading-[22px]">
        {prefix ? (
          <>
            <span className={`inline-flex items-center justify-center rounded px-1 mr-1.5 text-[10px] font-semibold font-mono uppercase tracking-wide leading-[16px] align-middle ${colors.badge}`}>
              {prefix.toUpperCase().slice(0, 5)}
            </span>
            {module && (
              <span className="text-zinc-600 mr-1 text-[11px]">{module}</span>
            )}
            <span className={`${colors.text} whitespace-pre-wrap break-words`}>{body}</span>
          </>
        ) : (
          <span className={`${colors.text} whitespace-pre-wrap break-words ${level === "trace" ? "font-mono text-[11px]" : ""}`}>{body}</span>
        )}
      </div>
    </div>
  )
}

export function LogViewer({ lines, isStreaming, className }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [search, setSearch] = useState("")
  const [searchOpen, setSearchOpen] = useState(false)
  const [filterLevel, setFilterLevel] = useState<"all" | "error" | "warn">("all")
  const searchRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [lines, autoScroll])

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current
    const atBottom = scrollHeight - scrollTop - clientHeight < 40
    setAutoScroll(atBottom)
  }, [])

  const classified = useMemo(() => {
    const base = lines.map((raw, i) => ({ raw, index: i, ...classifyLine(raw) }))
    return addGroupInfo(base)
  }, [lines])

  const filtered = useMemo(() => {
    let result = classified
    if (filterLevel === "error") {
      result = result.filter((l) => l.level === "error" || l.level === "trace" || l.level === "separator")
    } else if (filterLevel === "warn") {
      result = result.filter(
        (l) => l.level === "error" || l.level === "warn" || l.level === "trace" || l.level === "separator",
      )
    }
    if (search) {
      const q = search.toLowerCase()
      result = result.filter(
        (l) => l.body.toLowerCase().includes(q) || l.prefix.toLowerCase().includes(q) || l.module.toLowerCase().includes(q),
      )
    }
    return addGroupInfo(result)
  }, [classified, filterLevel, search])

  const errorCount = useMemo(() => classified.filter((l) => l.level === "error").length, [classified])
  const warnCount = useMemo(() => classified.filter((l) => l.level === "warn").length, [classified])

  const toggleSearch = useCallback(() => {
    setSearchOpen((prev) => {
      if (!prev) setTimeout(() => searchRef.current?.focus(), 0)
      else setSearch("")
      return !prev
    })
  }, [])

  return (
    <div className={`dark relative flex flex-col rounded-lg overflow-hidden border border-zinc-800/80 ${className ?? "h-[calc(80vh-220px)] min-h-[300px]"}`}>
      {/* Toolbar */}
      <div className="flex items-center gap-1.5 bg-[#111116] px-2 py-1 border-b border-zinc-800/80 shrink-0">
        <div className="flex items-center gap-0.5">
          <button
            onClick={() => setFilterLevel(filterLevel === "error" ? "all" : "error")}
            className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-mono font-medium transition-colors ${
              filterLevel === "error"
                ? "bg-[#f7768e]/15 text-[#f7768e]"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
            }`}
          >
            {errorCount > 0 && <span className="tabular-nums text-[10px]">{errorCount}</span>}
            errors
          </button>
          <button
            onClick={() => setFilterLevel(filterLevel === "warn" ? "all" : "warn")}
            className={`flex items-center gap-1 rounded-md px-2 py-1 text-[11px] font-mono font-medium transition-colors ${
              filterLevel === "warn"
                ? "bg-[#e0af68]/15 text-[#e0af68]"
                : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"
            }`}
          >
            {warnCount > 0 && <span className="tabular-nums text-[10px]">{warnCount}</span>}
            warnings
          </button>
        </div>

        <div className="flex-1" />

        {searchOpen && (
          <div className="flex items-center gap-1 bg-zinc-900 border border-zinc-700 rounded-md px-2 py-0.5 animate-message-in">
            <Search className="h-3 w-3 text-zinc-500 shrink-0" />
            <input
              ref={searchRef}
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="filter..."
              className="bg-transparent text-xs text-zinc-300 placeholder:text-zinc-600 focus:outline-none font-mono w-36"
            />
            {search && (
              <button onClick={() => setSearch("")} className="text-zinc-500 hover:text-zinc-300">
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        )}

        <button
          onClick={toggleSearch}
          className={`rounded-md p-1.5 transition-colors ${searchOpen ? "bg-zinc-800 text-zinc-200" : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50"}`}
        >
          <Search className="h-3.5 w-3.5" />
        </button>

        <span className="text-[11px] font-mono text-zinc-400 tabular-nums pl-1">
          {filtered.length !== lines.length ? `${filtered.length}/` : ""}{lines.length}
        </span>
      </div>

      {/* Log content */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 min-h-0 overflow-y-auto bg-[#0c0c12] py-1.5 px-1 font-mono text-xs"
        data-testid="log-viewer"
      >
        {filtered.map((item) => (
          <LogLine key={item.index} item={item} />
        ))}
        {isStreaming && (
          <div className="flex gap-0 pl-[48px] pt-1.5 pb-1">
            <span className="inline-block h-[14px] w-[2px] animate-pulse bg-[#6cb6ff]/60 rounded-full" />
          </div>
        )}
      </div>

      {!autoScroll && (
        <Button
          variant="secondary"
          size="sm"
          className="absolute bottom-3 right-3 h-7 gap-1 text-xs shadow-xl border border-zinc-700/80 bg-zinc-900/95 text-zinc-300 hover:bg-zinc-800 backdrop-blur-sm"
          onClick={() => {
            setAutoScroll(true)
            containerRef.current?.scrollTo({
              top: containerRef.current.scrollHeight,
              behavior: "smooth",
            })
          }}
        >
          <ArrowDown className="h-3 w-3" />
          Follow
        </Button>
      )}
    </div>
  )
}
