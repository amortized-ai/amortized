import type { MetricHistory } from "../api/use-jobs"

export const METRIC_ALIASES: Record<string, string[]> = {
  loss: ["loss", "train_loss", "avg_loss"],
  learning_rate: ["learning_rate", "lr"],
  grad_norm: ["grad_norm"],
  epoch: ["epoch"],
}

export const HISTORY_KEYS = [...new Set(Object.values(METRIC_ALIASES).flat())]

export function resolveMetric(
  values: Record<string, number>,
  canonical: string,
): number | undefined {
  const aliases = METRIC_ALIASES[canonical]
  if (!aliases) return values[canonical]
  for (const key of aliases) {
    if (key in values) return values[key]
  }
  return undefined
}

export function resolveHistory(
  histories: Record<string, MetricHistory[]>,
  canonical: string,
): MetricHistory[] | undefined {
  const aliases = METRIC_ALIASES[canonical]
  if (!aliases) return histories[canonical]
  for (const key of aliases) {
    if (histories[key]?.length) return histories[key]
  }
  return undefined
}
