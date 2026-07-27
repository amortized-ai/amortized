import { Badge } from "@/components/ui/badge"
import { useSystemHealth, type StatusLevel } from "@/hooks/use-system-health"
import { statusVariants } from "@/lib/status-variants"
import { Wifi, Loader2 } from "lucide-react"

const STATUS_ITEMS = [
  { key: "backend" as const, label: "Backend Connection", icon: Wifi },
] as const

function badgeClassName(level: StatusLevel): string {
  switch (level) {
    case "ok":
      return statusVariants.success
    case "error":
      return statusVariants.error
    case "loading":
      return ""
  }
}

function badgeLabel(level: StatusLevel): string {
  switch (level) {
    case "ok":
      return "Ready"
    case "error":
      return "Unavailable"
    case "loading":
      return "Checking…"
  }
}

export function SystemStatus() {
  const status = useSystemHealth()

  return (
    <div className="flex flex-wrap gap-3" data-testid="system-status">
      {STATUS_ITEMS.map((item) => {
        const level = status[item.key]
        const Icon = item.icon
        return (
          <Badge
            key={item.key}
            variant="outline"
            className={`flex items-center gap-1.5 px-3 py-1 ${badgeClassName(level)}`}
            data-testid={`status-${item.key}`}
          >
            {level === "loading" ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <Icon className="h-3.5 w-3.5" />
            )}
            <span>{item.label}</span>
            <span className="font-normal">— {badgeLabel(level)}</span>
          </Badge>
        )
      })}
    </div>
  )
}
