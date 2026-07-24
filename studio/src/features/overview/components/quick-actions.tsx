import { useNavigate } from "react-router"
import { Card, CardContent } from "@/components/ui/card"
import { Database, GraduationCap, FlaskConical, Rocket } from "lucide-react"

const ACTIONS = [
  {
    label: "Build a dataset",
    icon: Database,
    prompt: "Help me build a training dataset",
  },
  {
    label: "Train a model",
    icon: GraduationCap,
    prompt: "Help me train a model",
  },
  {
    label: "Evaluate a Model",
    icon: FlaskConical,
    prompt: "Help me evaluate a model",
  },
  {
    label: "Deploy a Model",
    icon: Rocket,
    prompt: "Help me deploy a model",
  },
] as const

export function QuickActions() {
  const navigate = useNavigate()

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {ACTIONS.map((action) => (
        <Card
          key={action.label}
          className="cursor-pointer transition-colors hover:bg-accent"
          onClick={() =>
            navigate(`/chat?prompt=${encodeURIComponent(action.prompt)}`)
          }
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault()
              navigate(`/chat?prompt=${encodeURIComponent(action.prompt)}`)
            }
          }}
        >
          <CardContent className="flex flex-col items-center gap-2 p-4 text-center">
            <action.icon className="h-6 w-6 text-muted-foreground" />
            <span className="text-sm font-medium">{action.label}</span>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
