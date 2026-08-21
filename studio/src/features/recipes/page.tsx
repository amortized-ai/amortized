import { useState, useMemo } from "react"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { ErrorState } from "@/components/error-state"
import { PageHeader } from "@/components/page-header"
import { TableSkeleton } from "@/components/table-skeleton"
import { SearchInput } from "@/components/search-input"
import { listDatasets } from "@/lib/api-client"
import { useQuery } from "@tanstack/react-query"
import { useJobRecipes, useStarterTemplates } from "./api/use-recipes"
import type { StarterTemplate } from "./api/use-recipes"
import { RecipeTable } from "./components/recipe-table"
import { RecipeDetailPanel } from "./components/recipe-detail-panel"
import { buildRecipeEntries, formatRecipeType, recipeTypeClassName } from "./lib/format"
import type { DatasetInfo } from "./lib/format"
import type { Job, JobType } from "@/types/api"
import { Beaker, ArrowRight } from "lucide-react"

function useDatasetMap() {
  return useQuery({
    queryKey: ["recipe-dataset-map"],
    queryFn: async () => {
      const items = await listDatasets()
      const map = new Map<string, DatasetInfo>()
      for (const d of items) {
        if (d.job_id) {
          map.set(d.job_id, { name: d.name, topic: d.topic, samples: d.samples })
        }
      }
      return map
    },
  })
}

interface SelectedConfig {
  job?: Job
  name: string
  config: Record<string, unknown>
  type: JobType
}

export default function RecipesPage() {
  const { data: jobs = [], isLoading, isError, error, refetch } = useJobRecipes()
  const { data: datasetMap = new Map() } = useDatasetMap()
  const { data: starterTemplates = [] } = useStarterTemplates()

  const [page, setPage] = useState(0)
  const [typeFilter, setTypeFilter] = useState<string>("all")
  const [search, setSearch] = useState("")
  const [selected, setSelected] = useState<SelectedConfig | null>(null)

  const recipes = useMemo(
    () => buildRecipeEntries(jobs, datasetMap),
    [jobs, datasetMap],
  )

  const countByType = useMemo(() => {
    const counts = { training: 0, sdg: 0 }
    for (const r of recipes) {
      const t = r.job.type as keyof typeof counts
      if (t in counts) counts[t]++
    }
    return counts
  }, [recipes])

  const filteredRecipes = useMemo(() => {
    let result = recipes
    if (typeFilter !== "all") {
      result = result.filter((r) => r.job.type === typeFilter)
    }
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter((r) => {
        return (
          r.name.toLowerCase().includes(q) ||
          r.description.toLowerCase().includes(q) ||
          r.job.id.toLowerCase().includes(q)
        )
      })
    }
    return result
  }, [recipes, typeFilter, search])

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  if (selected) {
    const job = selected.job ?? {
      id: "",
      type: selected.type,
      status: "succeeded" as const,
      config: selected.config,
      metadata: {},
      recipe: "",
      user_id: null,
      k8s_job_name: null,
      k8s_namespace: null,
      mlflow_run_id: null,
      mlflow_experiment: null,
      parent_job_id: null,
      error: null,
      created_at: new Date().toISOString(),
      started_at: null,
      completed_at: null,
    }
    return (
      <div className="h-[calc(100vh-4rem)]">
        <RecipeDetailPanel
          job={job}
          recipeName={selected.name}
          onBack={() => setSelected(null)}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Recipes"
        description="Reusable configs from previously successful jobs. Click a recipe to edit and re-execute."
      />

      <div className="grid gap-4 md:grid-cols-2">
        <button
          onClick={() => { setTypeFilter("sdg"); setPage(0) }}
          className={`group rounded-xl border bg-card p-4 text-left transition-all duration-200 hover:border-[#b6a6e9] hover:shadow-sm dark:hover:border-[#21134d] ${typeFilter === "sdg" ? "border-[#b6a6e9] shadow-sm" : ""}`}
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-sm">SDG Recipes</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Synthetic data generation configs</p>
            </div>
            <span className="text-2xl font-bold text-[#5e40be] dark:text-[#876fd4]">{countByType.sdg}</span>
          </div>
        </button>
        <button
          onClick={() => { setTypeFilter("training"); setPage(0) }}
          className={`group rounded-xl border bg-card p-4 text-left transition-all duration-200 hover:border-[#92c5f9] hover:shadow-sm dark:hover:border-[#003366] ${typeFilter === "training" ? "border-[#92c5f9] shadow-sm" : ""}`}
        >
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-semibold text-sm">Training Recipes</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Model fine-tuning configs</p>
            </div>
            <span className="text-2xl font-bold text-[#0066cc] dark:text-[#4394e5]">{countByType.training}</span>
          </div>
        </button>
      </div>

      <div className="flex items-center justify-between gap-4">
        <Tabs value={typeFilter} onValueChange={(v) => { setTypeFilter(v); setPage(0) }}>
          <TabsList>
            <TabsTrigger value="all">All ({recipes.length})</TabsTrigger>
            <TabsTrigger value="sdg">SDG ({countByType.sdg})</TabsTrigger>
            <TabsTrigger value="training">Training ({countByType.training})</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">{filteredRecipes.length} visible</span>
          <SearchInput
            value={search}
            onChange={(v) => { setSearch(v); setPage(0) }}
            placeholder="Search recipes..."
          />
        </div>
      </div>

      <div className="max-h-[50vh] overflow-y-auto rounded-lg border">
        {isLoading ? (
          <TableSkeleton columns={4} />
        ) : (
          <RecipeTable
            recipes={filteredRecipes}
            page={page}
            onPageChange={setPage}
            onSelectJob={(job, name) =>
              setSelected({ job, name, config: job.config, type: job.type as JobType })
            }
          />
        )}
      </div>

      {/* Starter Templates */}
      {starterTemplates.length > 0 && (
        <div className="space-y-3 pt-4 border-t">
          <div>
            <h2 className="text-lg font-semibold">Starter Templates</h2>
            <p className="text-sm text-muted-foreground">
              Curated configs tested by researchers. Use as a starting point for new jobs.
            </p>
          </div>
          <div className="flex gap-3 overflow-x-auto pb-2">
            {starterTemplates.map((t) => (
              <StarterTemplateCard
                key={`${t.type}-${t.use_case}`}
                template={t}
                onSelect={() =>
                  setSelected({
                    name: t.name,
                    config: t.config,
                    type: t.type as JobType,
                  })
                }
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function StarterTemplateCard({
  template,
  onSelect,
}: {
  template: StarterTemplate
  onSelect: () => void
}) {
  return (
    <button
      onClick={onSelect}
      className="group rounded-xl border bg-card p-4 text-left transition-all duration-200 hover:border-primary/30 hover:shadow-sm w-[320px] shrink-0"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2.5">
          <Beaker className="h-4 w-4 text-muted-foreground shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium">{template.name}</p>
            {template.description && (
              <p className="text-xs text-muted-foreground mt-0.5">{template.description}</p>
            )}
            <Badge className={`mt-2 ${recipeTypeClassName(template.type)}`}>
              {formatRecipeType(template.type)}
            </Badge>
          </div>
        </div>
        <ArrowRight className="h-4 w-4 text-muted-foreground/0 transition-all duration-200 group-hover:text-muted-foreground group-hover:translate-x-0.5 shrink-0 mt-0.5" />
      </div>
    </button>
  )
}
