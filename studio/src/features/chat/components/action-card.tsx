import { useState } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ShieldCheck, Loader2 } from "lucide-react"
import type { ProposedAction } from "../types"

interface ActionCardProps {
  action: ProposedAction
  onConfirm: () => void
  onReject: () => void
}

export function ActionCard({ action, onConfirm, onReject }: ActionCardProps) {
  const [submitting, setSubmitting] = useState(false)

  const handleConfirm = () => {
    setSubmitting(true)
    onConfirm()
  }

  return (
    <Card className="border-primary/30 bg-primary/5">
      <CardContent className="p-4">
        <div className="flex items-start gap-2">
          <ShieldCheck className="mt-0.5 h-4 w-4 text-primary" />
          <div className="flex-1">
            <CardTitle className="text-sm">{action.action}</CardTitle>
            <CardDescription className="mt-1 text-xs">
              {action.description}
            </CardDescription>
            {action.config && (
              <div className="mt-2 rounded-md bg-muted/50 p-2 text-xs font-mono max-h-48 overflow-y-auto space-y-0.5">
                {Object.entries(action.config).map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <span className="text-muted-foreground shrink-0">{key}:</span>
                    <span className="break-all">
                      {typeof value === "object" ? JSON.stringify(value) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </CardContent>
      <CardFooter className="gap-2 p-4 pt-0">
        <Button size="sm" onClick={handleConfirm} disabled={submitting}>
          {submitting ? (
            <>
              <Loader2 className="mr-1 h-3 w-3 animate-spin" />
              Submitting
            </>
          ) : (
            "Confirm"
          )}
        </Button>
        <Button size="sm" variant="outline" onClick={onReject} disabled={submitting}>
          Cancel
        </Button>
      </CardFooter>
    </Card>
  )
}
