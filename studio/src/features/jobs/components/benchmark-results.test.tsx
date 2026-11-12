import { render, screen, within } from "@testing-library/react"
import { describe, it, expect, vi } from "vitest"
import { MemoryRouter } from "react-router"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import fixture from "@/test/fixtures/evalhub-results.json"
import type { EvalResults, EvaluationGroup } from "@/lib/api-client"
import { extractBenchmarkView } from "@/features/chat/utils/parse-tool-result"
import { EvaluationDetailPanel } from "@/features/evaluation/components/evaluation-detail-panel"
import { ActionCard } from "@/features/chat/components/action-card"
import { BenchmarkResults } from "./benchmark-results"
import { formatBenchmarkValue, isRunAverage } from "../lib/format"

vi.mock("@/features/models/api/use-models", () => ({ useModels: () => ({ data: [] }) }))
vi.mock("@/features/datasets/api/use-datasets", () => ({
  useDatasets: () => ({ data: [] }),
  fetchDatasetByRun: async () => null,
}))

// Each entry is one benchmark's real API payload from its conformance run
// (`make evalhub-contract` writes the fixture): the card the user approves and the
// results the job page shows. Studio must show the contract's score for every entry.

interface FixtureEntry {
  provider_id: string
  id: string
  card: Record<string, unknown>
  results: EvalResults
}

const entries = (fixture as unknown as { entries: FixtureEntry[] }).entries

const rowsOf = (element: HTMLElement) =>
  within(element).getAllByRole("row") as HTMLTableRowElement[]

describe.each(entries.map((e) => [`${e.provider_id} ${e.id}`, e] as const))(
  "benchmark %s",
  (_, entry) => {
    const results = entry.results
    const primary = results.primary_metric ?? null
    const all = Object.keys(results.scores ?? {})
    const reported = all.filter((name) => !isRunAverage(name, primary))

    it("the card names the primary metric the results show", () => {
      const card = extractBenchmarkView(entry.card)!
      expect(card.metric).toBe(primary ?? "")
      if (!primary) expect(card.noScore).toBe(results.no_score_reason)
      render(
        <QueryClientProvider client={new QueryClient()}>
          <ActionCard
            action={{ action: "create_eval_job", description: "", params: {}, jobType: "eval", benchmark: card }}
            onConfirm={() => {}}
            onReject={() => {}}
          />
        </QueryClientProvider>,
      )
      const row = screen.getByText("Primary metric").closest("tr")!
      expect(row).toHaveTextContent(primary ? `Primary metric${primary}` : `Primary metricnone: ${card.noScore}`)
    })

    it("the results tab leads with the primary metric, or says why there is none", () => {
      render(<BenchmarkResults results={results} />)
      if (primary) {
        const headline = screen.getByTestId("benchmark-score")
        expect(headline).toHaveTextContent(`Primary metric${primary}`)
        expect(headline).toHaveTextContent(formatBenchmarkValue(results.primary_score!))
        const direction = results.primary_lower_is_better ? "lower" : "higher"
        expect(headline).toHaveTextContent(`${direction} is better`)
        expect(screen.queryByTestId("benchmark-no-score")).toBeNull()
      } else {
        expect(screen.getByTestId("benchmark-no-score")).toHaveTextContent(
          results.no_score_reason!,
        )
        expect(screen.queryByTestId("benchmark-score")).toBeNull()
      }
    })

    it("lists every other metric once, stderr as ± beside it, with its direction; no run average", () => {
      render(<BenchmarkResults results={results} />)
      // The score is in its card only; the table holds the rest, or is absent.
      const others = reported.filter((name) => name !== primary)
      const table = screen.queryByTestId("benchmark-metrics")
      if (others.length === 0) {
        expect(table).toBeNull()
        return
      }
      const rows = rowsOf(table!).slice(1)
      expect(rows.map((row) => row.cells[0]!.textContent)).toHaveLength(others.length)
      others.forEach((name, i) => {
        const [label, value] = [rows[i]!.cells[0]!, rows[i]!.cells[1]!]
        expect(label.textContent).toContain(name)
        expect(value).toHaveTextContent(formatBenchmarkValue(results.scores![name]!))
        const stderr = results.stderr?.[name]
        if (stderr != null) expect(value).toHaveTextContent(`±${formatBenchmarkValue(stderr)}`)
        const direction = results.metric_metadata?.[name]?.direction
        if (direction) expect(label).toHaveTextContent(`${direction} is better`)
      })
      for (const name of Object.keys(results.stderr ?? {})) {
        expect(rows.some((row) => row.cells[0]!.textContent?.startsWith(`${name}_stderr`))).toBe(false)
      }
      expect(rows.some((row) => row.cells[0]!.textContent?.startsWith("all."))).toBe(primary?.startsWith("all.") ?? false)
    })

    it("the comparison marks the primary metric's row first", () => {
      if (!primary) return
      const group: EvaluationGroup = {
        id: "group-1",
        dataset: { run_id: "", name: entry.id },
        metric_names: all,
        metric_metadata: results.metric_metadata,
        primary_metric: primary,
        evals: [
          {
            job_id: "11111111-2222-3333-4444-555555555555",
            model: "base-model",
            status: "succeeded",
            created_at: "2026-11-09T00:00:00Z",
            mlflow_run_id: "",
            scores: results.scores as Record<string, number>,
            num_samples: 10,
            topic: "",
          },
        ],
        latest_created_at: "2026-11-09T00:00:00Z",
      }
      render(
        <QueryClientProvider client={new QueryClient()}>
          <MemoryRouter>
            <EvaluationDetailPanel group={group} open onOpenChange={() => {}} />
          </MemoryRouter>
        </QueryClientProvider>,
      )
      const body = screen.getAllByRole("rowgroup").at(-1)!
      const rows = rowsOf(body)
      expect(rows[0]!.cells[0]).toHaveTextContent(primary)
      expect(rows[0]!.cells[0]).toHaveTextContent("primary")
      expect(rows).toHaveLength(reported.length)
    })
  },
)

// No entry on the pinned images ran without a score, so the fixture has none.
it("a benchmark without a score says why and still lists its metrics", () => {
  const reason =
    "the catalog's metric is acc; the adapter reports bits_per_byte, byte_perplexity, word_perplexity"
  const results = {
    kind: "benchmark",
    scores: { word_perplexity: 30, bits_per_byte: 0.9 },
    metric_metadata: { word_perplexity: { direction: "lower" } },
    primary_metric: null,
    no_score_reason: reason,
  } as unknown as EvalResults
  render(<BenchmarkResults results={results} />)
  expect(screen.getByTestId("benchmark-no-score")).toHaveTextContent(`No primary metric${reason}`)
  expect(screen.queryByTestId("benchmark-score")).toBeNull()
  const rows = rowsOf(screen.getByTestId("benchmark-metrics")).slice(1)
  expect(rows.map((row) => row.cells[0]!.textContent)).toEqual([
    "word_perplexitylower is better",
    "bits_per_byte",
  ])
})

// LightEval's "all.<m>" averages the run's tasks; a job runs one, so it repeats the score.
it("hides the run average unless it is the score itself", () => {
  const one = {
    kind: "benchmark",
    scores: { "all.extractive_match": 0.1, "gsm8k.extractive_match": 0.1 },
    primary_metric: "gsm8k.extractive_match",
    primary_score: 0.1,
  } as unknown as EvalResults
  const { unmount } = render(<BenchmarkResults results={one} />)
  expect(screen.getByTestId("benchmark-score")).toHaveTextContent("gsm8k.extractive_match")
  // Its only other metric is the run average, so there is no table to repeat the score.
  expect(screen.queryByTestId("benchmark-metrics")).toBeNull()
  unmount()
  const suite = { ...one, primary_metric: "all.extractive_match" } as unknown as EvalResults
  render(<BenchmarkResults results={suite} />)
  expect(rowsOf(screen.getByTestId("benchmark-metrics")).slice(1).map((r) => r.cells[0]!.textContent)).toEqual([
    "gsm8k.extractive_match",
  ])
})

it("the fixture covers verified entries of both providers", () => {
  expect(new Set(entries.map((e) => e.provider_id))).toEqual(
    new Set(["amortized-lm-eval", "amortized-lighteval"]),
  )
})
