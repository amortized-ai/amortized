import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { CheckCircle2 } from "lucide-react"

interface ExecuteDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  config: Record<string, unknown>
  onConfirm: () => void
  isPending: boolean
  submittedJobId?: string | null
  onViewJob?: () => void
}

export function ExecuteDialog({
  open,
  onOpenChange,
  config,
  onConfirm,
  isPending,
  submittedJobId,
  onViewJob,
}: ExecuteDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        {submittedJobId ? (
          <>
            <DialogHeader>
              <div className="flex flex-col items-center gap-3 pt-2">
                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-rh-green-light/50 text-rh-green dark:bg-rh-green-dark/40 dark:text-rh-green">
                  <CheckCircle2 className="h-6 w-6" />
                </div>
                <DialogTitle>Job Submitted</DialogTitle>
                <DialogDescription className="text-center">
                  Your job is now queued and will begin processing shortly.
                </DialogDescription>
              </div>
            </DialogHeader>
            <div className="rounded-lg bg-muted/50 px-4 py-3">
              <p className="text-xs text-muted-foreground mb-1">Job ID</p>
              <p className="text-sm font-mono break-all">{submittedJobId}</p>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Back to Recipe
              </Button>
              <Button onClick={onViewJob}>
                View Job
              </Button>
            </DialogFooter>
          </>
        ) : (
          <>
            <DialogHeader>
              <DialogTitle>Execute Recipe</DialogTitle>
              <DialogDescription>
                Review the configuration before submitting the job.
              </DialogDescription>
            </DialogHeader>
            <ScrollArea className="max-h-[40vh]">
              <pre className="rounded-md bg-muted p-4 text-xs" data-testid="execute-config-preview">
                {JSON.stringify(config, null, 2)}
              </pre>
            </ScrollArea>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={isPending}
              >
                Cancel
              </Button>
              <Button
                onClick={onConfirm}
                disabled={isPending}
                data-testid="confirm-execute"
              >
                {isPending ? "Submitting..." : "Submit Job"}
              </Button>
            </DialogFooter>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}
