import { useState } from "react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface SaveDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSave: (name: string) => void
  defaultName: string
}

export function SaveDialog({
  open,
  onOpenChange,
  onSave,
  defaultName,
}: SaveDialogProps) {
  const [name, setName] = useState(defaultName)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Save Recipe As</DialogTitle>
          <DialogDescription>
            Enter a name for the new recipe.
          </DialogDescription>
        </DialogHeader>
        <Input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Recipe name"
          data-testid="save-as-name-input"
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              onSave(name)
              onOpenChange(false)
            }}
            disabled={!name.trim()}
            data-testid="confirm-save-as"
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
