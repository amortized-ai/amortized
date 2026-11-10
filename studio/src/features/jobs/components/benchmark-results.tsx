import type { EvalResults } from "@/lib/api-client"
import { formatBenchmarkValue, isRunAverage } from "../lib/format"

// A benchmark job's results, as the server presents them: the contract's primary metric
// first, then every other metric the adapter reported, once, with its stderr as ± and its
// direction. Studio picks nothing itself — which metric is primary comes from the API.

function better(direction: string | null | undefined): string {
  return direction ? `${direction} is better` : ""
}

export function BenchmarkResults({ results, modelName }: { results: EvalResults; modelName?: string }) {
  const primary = results.primary_metric ?? null
  const scores = Object.entries(results.scores ?? {}).filter(
    (entry): entry is [string, number] =>
      typeof entry[1] === "number" && !isRunAverage(entry[0], primary),
  )
  const primaryScore = results.primary_score
  const headline = primary !== null && typeof primaryScore === "number"
  // The primary metric is shown once, in its card; the table holds the rest.
  const others = headline ? scores.filter(([name]) => name !== primary) : scores
  const metadata = results.metric_metadata ?? {}
  const direction = primary
    ? results.primary_lower_is_better
      ? "lower"
      : "higher"
    : null

  return (
    <div className="space-y-4 pt-2">
      {modelName ? (
        <p className="text-sm text-muted-foreground">
          Model: <span className="font-mono text-foreground">{modelName}</span>
        </p>
      ) : null}

      {headline ? (
        <div className="rounded-xl border bg-card p-4" data-testid="benchmark-score">
          <p className="text-xs text-muted-foreground">Primary metric</p>
          <p className="mt-1 text-lg">
            <span className="font-mono">{primary}</span>{" "}
            <span className="font-semibold tabular-nums">{formatBenchmarkValue(primaryScore)}</span>
            {results.stderr?.[primary] != null && (
              <span className="ml-1 text-sm text-muted-foreground tabular-nums">
                ±{formatBenchmarkValue(results.stderr[primary]!)}
              </span>
            )}
          </p>
          <p className="text-xs text-muted-foreground">{better(direction)}</p>
        </div>
      ) : (
        <div
          className="rounded-xl border border-amber-500/40 bg-amber-500/10 p-4 text-sm"
          data-testid="benchmark-no-score"
        >
          <p className="font-medium">No primary metric</p>
          <p className="text-muted-foreground mt-1">
            {results.no_score_reason || "The adapter reported no metrics."}
          </p>
        </div>
      )}

      {others.length > 0 && (
        <div className="rounded-xl border bg-card overflow-hidden">
          <table className="w-full text-sm" data-testid="benchmark-metrics">
            <thead>
              <tr className="border-b bg-muted/40">
                <th className="px-4 py-2.5 text-left font-medium">
                  {headline ? "Other metrics" : "Metric"}
                </th>
                <th className="px-4 py-2.5 text-right font-medium">Value</th>
              </tr>
            </thead>
            <tbody>
              {others.map(([name, value]) => (
                <tr key={name} className="border-b last:border-0 border-border/40">
                  <td className="px-4 py-2.5 font-mono text-xs">
                    {name}
                    {metadata[name]?.direction && (
                      <span className="block font-sans text-muted-foreground">
                        {better(metadata[name]!.direction)}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-2.5 text-right font-mono text-xs tabular-nums">
                    {formatBenchmarkValue(value)}
                    {results.stderr?.[name] != null && (
                      <span className="ml-1 text-muted-foreground">
                        ±{formatBenchmarkValue(results.stderr[name]!)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
