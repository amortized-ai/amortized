import JsonView from "react18-json-view"
import "react18-json-view/src/style.css"

interface JsonTreeViewerProps {
  data: unknown
  collapsed?: number
  className?: string
}

export function JsonTreeViewer({
  data,
  collapsed = 2,
  className,
}: JsonTreeViewerProps) {
  return (
    <div className={className} data-testid="json-tree-viewer">
      <JsonView
        src={data}
        collapsed={collapsed}
        enableClipboard
        theme="default"
        style={{ fontSize: "0.75rem", fontFamily: "var(--font-mono, monospace)" }}
      />
    </div>
  )
}
