import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable"
import type { ReactNode } from "react"

interface ResizableLayoutProps {
  left: ReactNode
  right: ReactNode
  defaultLeftSize?: number
  minLeftSize?: number
  minRightSize?: number
  direction?: "horizontal" | "vertical"
}

export function ResizableLayout({
  left,
  right,
  defaultLeftSize = 60,
  minLeftSize = 30,
  minRightSize = 20,
  direction = "horizontal",
}: ResizableLayoutProps) {
  return (
    <ResizablePanelGroup orientation={direction} className="min-h-[400px]">
      <ResizablePanel defaultSize={defaultLeftSize} minSize={minLeftSize}>
        {left}
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel minSize={minRightSize}>
        {right}
      </ResizablePanel>
    </ResizablePanelGroup>
  )
}
