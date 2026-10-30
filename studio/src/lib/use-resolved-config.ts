import { useMemo } from "react"
import { useQuery } from "@tanstack/react-query"

import { resolvePrompts } from "@/lib/api-client"
import { findPromptRefs, resolvePromptRefs } from "@/features/recipes/lib/placeholders"

// Prompt store (#541): a config stores prompts as `<prompt:...>` reference tokens.
// Any user-visible surface that renders a config must show the resolved prompt
// TEXT, never the raw token. This hook resolves every ref for DISPLAY and — unlike
// the bare `resolvePromptRefs` — guarantees no token survives: a ref that is still
// loading or can't be resolved (resolve failed, or the row is gone) renders as a
// readable placeholder instead of leaking `<prompt:prm_...>`.
//
// Read-only: it returns a resolved COPY and never mutates the source config (the
// persisted config stays ref-form). For the editable recipe form, which hydrates
// ref text back INTO editable state, see recipe-detail-panel's own resolve.
export function useResolvedConfig<T>(config: T): T {
  const refIds = useMemo(() => findPromptRefs(config as unknown), [config])
  const { data, isError } = useQuery({
    queryKey: ["resolve-prompts", refIds],
    queryFn: () => resolvePrompts(refIds),
    enabled: refIds.length > 0,
    staleTime: Infinity,
  })
  return useMemo(() => {
    if (refIds.length === 0) return config
    const resolved = data?.prompts
    const texts: Record<string, string> = {}
    for (const id of refIds) {
      const text = resolved?.[id]
      if (text !== undefined) {
        texts[id] = text
      } else if (!resolved && !isError) {
        texts[id] = "(loading prompt…)"
      } else {
        texts[id] = `(prompt text unavailable: ${id})`
      }
    }
    return resolvePromptRefs(config, texts)
  }, [config, data, isError, refIds])
}
