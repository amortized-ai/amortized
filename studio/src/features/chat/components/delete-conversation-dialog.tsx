import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Trash2, AlertTriangle } from "lucide-react"

interface DeleteConversationDialogProps {
  open: boolean
  conversationTitle: string
  onConfirm: () => void
  onCancel: () => void
}

export function DeleteConversationDialog({
  open,
  conversationTitle,
  onConfirm,
  onCancel,
}: DeleteConversationDialogProps) {
  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && onCancel()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-rh-danger-light/50 dark:bg-rh-danger-dark/20">
              <AlertTriangle className="h-5 w-5 text-rh-danger dark:text-rh-danger" />
            </div>
            <div className="flex-1">
              <DialogTitle className="text-lg font-semibold">Delete conversation?</DialogTitle>
              <DialogDescription className="mt-2 text-sm leading-relaxed">
                This will permanently delete{" "}
                <span className="font-medium text-foreground">"{conversationTitle}"</span> and all
                its messages. This action cannot be undone.
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>
        <DialogFooter className="gap-2 sm:gap-2">
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={onConfirm}
            className="gap-2"
          >
            <Trash2 className="h-4 w-4" />
            Delete conversation
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
