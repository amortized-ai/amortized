import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ShieldCheck } from "lucide-react"
import type { ProposedAction } from "../types"

interface ActionCardProps {
  action: ProposedAction
  onConfirm: () => void
  onReject: () => void
}

export function ActionCard({ action, onConfirm, onReject }: ActionCardProps) {
  return (
    <Card className="border-primary/30 bg-primary/5">
      <CardContent className="p-4">
        <div className="flex items-start gap-2">
          <ShieldCheck className="mt-0.5 h-4 w-4 text-primary" />
          <div>
            <CardTitle className="text-sm">{action.action}</CardTitle>
            <CardDescription className="mt-1 text-xs">
              {action.description}
            </CardDescription>
          </div>
        </div>
      </CardContent>
      <CardFooter className="gap-2 p-4 pt-0">
        <Button size="sm" onClick={onConfirm}>
          Confirm
        </Button>
        <Button size="sm" variant="outline" onClick={onReject}>
          Reject
        </Button>
      </CardFooter>
    </Card>
  )
}
