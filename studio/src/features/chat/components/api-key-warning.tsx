import { Link } from "react-router"
import { ShieldAlert } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useModelProvider } from "@/features/settings/api/use-model-provider"

// Shown when the chat box holds what looks like an API key: anything sent in chat reaches the
// chat model and stays in the chat history, so keys belong in Settings (Model Providers).
export function ApiKeyWarning({ onRemove, onSendAnyway }: { onRemove: () => void; onSendAnyway: () => void }) {
  const { data } = useModelProvider()

  return (
    <div
      role="alert"
      className="rounded-lg border border-[#f0ab00] bg-[#fdf7e7] px-3 py-2 text-xs text-[#795600] dark:border-[#73480b] dark:bg-[#3d2c00]/40 dark:text-[#ffcc17]"
    >
      <div className="flex items-start gap-2">
        <ShieldAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <p>
          This message looks like it contains an API key. Anything sent in chat goes to the chat model
          and stays in your chat history.{" "}
          {data?.available ? (
            <>
              Add the model&apos;s provider under{" "}
              <Link to="/settings" className="font-medium underline underline-offset-2">
                Settings
              </Link>{" "}
              instead (MaaS for your own OpenAI-compatible endpoint).
            </>
          ) : (
            "Ask your administrator to configure the model's provider instead."
          )}
        </p>
      </div>
      <div className="mt-2 flex justify-end gap-2">
        <Button size="sm" variant="ghost" className="h-7 text-xs" onClick={onSendAnyway}>
          Send anyway
        </Button>
        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={onRemove}>
          Remove the key
        </Button>
      </div>
    </div>
  )
}
