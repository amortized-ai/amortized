import CodeMirror from "@uiw/react-codemirror"
import { json, jsonParseLinter } from "@codemirror/lang-json"
import { linter } from "@codemirror/lint"
import { ScrollArea } from "@/components/ui/scroll-area"

interface JsonEditorInnerProps {
  jsonValue: string
  jsonError: string | null
  schema: Record<string, unknown> | null
  onJsonChange: (value: string) => void
}

function SchemaOutline({ schema }: { schema: Record<string, unknown> }) {
  const properties = (schema.properties ?? {}) as Record<
    string,
    { type?: string; description?: string }
  >
  const entries = Object.entries(properties)

  if (entries.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No schema available</p>
    )
  }

  return (
    <div className="space-y-3">
      {entries.map(([key, prop]) => (
        <div key={key} className="space-y-0.5">
          <div className="flex items-center gap-2">
            <code className="text-sm font-medium text-foreground">{key}</code>
            {prop.type && (
              <span className="text-xs text-muted-foreground">
                ({prop.type})
              </span>
            )}
          </div>
          {prop.description && (
            <p className="text-xs text-muted-foreground">{prop.description}</p>
          )}
        </div>
      ))}
    </div>
  )
}

export default function JsonEditorInner({
  jsonValue,
  jsonError,
  schema,
  onJsonChange,
}: JsonEditorInnerProps) {
  return (
    <div className="flex gap-4 h-full overflow-hidden" data-testid="json-editor-panes">
      <div className="w-1/4 flex flex-col overflow-hidden border-r pr-4">
        <h3 className="mb-3 text-sm font-semibold shrink-0">Schema</h3>
        <ScrollArea className="flex-1">
          {schema ? (
            <SchemaOutline schema={schema} />
          ) : (
            <p className="text-sm text-muted-foreground">No schema loaded</p>
          )}
        </ScrollArea>
      </div>

      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center justify-between mb-2 shrink-0">
          <h3 className="text-sm font-semibold">JSON Configuration</h3>
          {jsonError && (
            <span className="text-xs text-destructive" data-testid="json-error">
              {jsonError}
            </span>
          )}
        </div>
        <div className="flex-1 overflow-hidden rounded-md border">
          <CodeMirror
            value={jsonValue}
            extensions={[json(), linter(jsonParseLinter())]}
            onChange={onJsonChange}
            data-testid="codemirror-editor"
            className="h-full [&_.cm-editor]:h-full [&_.cm-scroller]:overflow-auto"
            basicSetup={{
              lineNumbers: true,
              foldGutter: true,
              bracketMatching: true,
              closeBrackets: true,
            }}
          />
        </div>
      </div>
    </div>
  )
}

export type { JsonEditorInnerProps }
export { SchemaOutline }
