import { Link } from "react-router"
import { ShieldAlert } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useCustomEndpoints } from "@/features/settings/api/use-custom-endpoints"
import { CUSTOM_ENDPOINTS_SECTION_ID } from "@/features/settings/components/custom-endpoints-card"

// Shown when the chat box holds what looks like an API key: anything sent in chat reaches the
// chat model and stays in the chat history, so keys belong in Settings > Custom endpoints.
export function ApiKeyWarning({ onRemove, onSendAnyway }: { onRemove: () => void; onSendAnyway: () => void }) {
  const { data } = useCustomEndpoints()

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
              Add the endpoint under{" "}
              <Link
                to={{ pathname: "/settings", hash: `#${CUSTOM_ENDPOINTS_SECTION_ID}` }}
                className="font-medium underline underline-offset-2"
              >
                Settings &gt; Custom endpoints
              </Link>{" "}
              and give Morty its name instead.
            </>
          ) : (
            "Ask your administrator to configure the endpoint, and give Morty its name instead."
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
