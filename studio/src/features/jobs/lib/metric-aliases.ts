import type { MetricHistory } from "../api/use-jobs"

export const METRIC_ALIASES: Record<string, string[]> = {
  loss: ["loss", "train_loss", "avg_loss"],
  learning_rate: ["learning_rate", "lr"],
  grad_norm: ["grad_norm"],
  epoch: ["epoch"],
}

export const HISTORY_KEYS = [...new Set(Object.values(METRIC_ALIASES).flat())]

/**
 * verl (the lora_grpo backend) namespaces its metrics (``actor/…``, ``critic/…``), so none of
 * the SFT aliases above match. These keys are only fetched when the run actually logged them,
 * i.e. only for GRPO runs.
 */
export const GRPO_METRICS = {
  reward: "critic/rewards/mean",
  entropy: "actor/entropy",
  grad_norm: "actor/grad_norm",
  learning_rate: "actor/lr",
  global_step: "training/global_step",
} as const

export const GRPO_HISTORY_KEYS = [
  GRPO_METRICS.reward,
  GRPO_METRICS.entropy,
  GRPO_METRICS.grad_norm,
  GRPO_METRICS.learning_rate,
]

/** Every metric key whose step history the metrics views fetch (when present on the run). */
export const ALL_HISTORY_KEYS = [...HISTORY_KEYS, ...GRPO_HISTORY_KEYS]

/** True for a GRPO training job's ``config.algorithm`` (``lora_grpo`` / ``lora-grpo`` / ``grpo``). */
export function isGrpoAlgorithm(algorithm: unknown): boolean {
  if (typeof algorithm !== "string") return false
  const algo = algorithm.trim().toLowerCase().replace(/-/g, "_")
  return algo === "lora_grpo" || algo === "grpo"
}

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
