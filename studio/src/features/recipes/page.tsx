import { useState, useCallback, useMemo } from "react"
import { useNavigate } from "react-router"
import { Link } from "react-router"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ErrorState } from "@/components/error-state"
import { PageHeader } from "@/components/page-header"
import { TableSkeleton } from "@/components/table-skeleton"
import { SearchInput } from "@/components/search-input"
import { ArrowLeft, Play, Save, Copy, GraduationCap, Sparkles, ArrowRight, Lightbulb } from "lucide-react"
import { useRecipes, useExecuteRecipe, useSaveRecipe } from "./api/use-recipes"
import { useRecipeState } from "./hooks/use-recipe-state"
import { RecipeTable } from "./components/recipe-table"
import { RecipeBuilderForm } from "./components/recipe-builder-form"
import { JsonEditorDialog } from "./components/json-editor-dialog"
import { ExecuteDialog } from "./components/execute-dialog"
import { SaveDialog } from "./components/save-dialog"
import { useDatasets } from "@/features/datasets/api/use-datasets"
import { useModels } from "@/features/models/api/use-models"
import type { Recipe } from "@/types/api"
import { isUsefulRecipe, getEffectiveType } from "./lib/format"

const RECIPE_GUIDES: Record<string, { color: string; iconBg: string; steps: string[] }> = {
  sdg: {
    color: "text-[#5e40be] dark:text-[#876fd4]",
    iconBg: "bg-[#ece6ff] dark:bg-[#1b0d33]/40",
    steps: [
      "Set the teacher model — use a local model name (e.g. granite-3.1-8b) with api_base pointed at your serving endpoint, or a cloud provider in provider/model format",
      "Choose how many samples to generate (start with 50–100 to test)",
      "Input data is optional — most SDG recipes generate from scratch",
      "Click Execute to submit the job, then monitor it on the Jobs tab",
    ],
  },
  training: {
    color: "text-[#0066cc] dark:text-[#4394e5]",
    iconBg: "bg-[#e0f0ff] dark:bg-[#003366]/40",
    steps: [
      "Pick a training method — LoRA SFT is the fastest and most memory-efficient",
      "Select a base model from the registry (must be registered in MLflow first)",
      "Select a training dataset (run an SDG recipe first if none are available)",
      "Adjust hyperparameters or keep defaults, then click Execute",
    ],
  },
}

function RecipeBuilderGuide({ type, hasDatasets, hasModels }: { type: string; hasDatasets: boolean; hasModels: boolean }) {
  const guide = RECIPE_GUIDES[type]
  if (!guide) return null

  const warnings: string[] = []
  if (type === "training" && !hasDatasets) warnings.push("No datasets available yet — run an SDG recipe first to generate training data.")
  if (type === "training" && !hasModels) warnings.push("No base models found — register a model in MLflow (e.g. a Granite or Qwen model from HuggingFace).")
  return (
    <div className="mx-auto max-w-2xl animate-message-in rounded-xl border bg-card p-4 space-y-3">
      <div className="flex items-center gap-2">
        <div className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md ${guide.iconBg} ${guide.color}`}>
          <Lightbulb className="h-3.5 w-3.5" />
        </div>
        <p className="text-sm font-medium">How to configure this recipe</p>
      </div>
      <ol className="space-y-1.5 pl-8">
        {guide.steps.map((step, i) => (
          <li key={i} className="text-xs text-muted-foreground list-decimal">{step}</li>
        ))}
      </ol>
      {warnings.length > 0 && (
        <div className="border-t pt-3 space-y-1.5">
          {warnings.map((w, i) => (
            <p key={i} className="text-xs text-[#b98412] dark:text-[#dca614]">{w}</p>
          ))}
        </div>
      )}
    </div>
  )
}

export default function RecipesPage() {
  const navigate = useNavigate()
  const { data: recipes = [], isLoading, isError, error, refetch } = useRecipes()
  const { data: datasets = [] } = useDatasets()
  const { data: models = [] } = useModels()
  const executeMutation = useExecuteRecipe()
  const saveMutation = useSaveRecipe()

  const [page, setPage] = useState(0)
  const [typeFilter, setTypeFilter] = useState<string>("all")
  const [search, setSearch] = useState("")
  const [builderOpen, setBuilderOpen] = useState(false)
  const [selectedRecipe, setSelectedRecipe] = useState<Recipe | null>(null)
  const [jsonEditorOpen, setJsonEditorOpen] = useState(false)
  const [executeDialogOpen, setExecuteDialogOpen] = useState(false)
  const [submittedJobId, setSubmittedJobId] = useState<string | null>(null)
  const [saveDialogOpen, setSaveDialogOpen] = useState(false)

  const { state, setField, setJson, loadRecipe, reset, getConfig } =
    useRecipeState()

  const visibleRecipes = useMemo(() => recipes.filter(isUsefulRecipe), [recipes])

  const filteredRecipes = useMemo(() => {
    let result = visibleRecipes
    if (typeFilter !== "all") {
      result = result.filter((r) => getEffectiveType(r).toLowerCase() === typeFilter)
    }
    if (search.trim()) {
      const q = search.toLowerCase()
      result = result.filter((r) =>
        r.name.toLowerCase().includes(q) || (r.description ?? "").toLowerCase().includes(q)
      )
    }
    return result
  }, [visibleRecipes, typeFilter, search])

  const recipeCountByType = useMemo(() => {
    const counts = { training: 0, sdg: 0 }
    for (const r of visibleRecipes) {
      const t = getEffectiveType(r).toLowerCase() as keyof typeof counts
      if (t in counts) counts[t]++
    }
    return counts
  }, [visibleRecipes])

  const handleSelectRecipe = useCallback(
    (recipe: Recipe) => {
      loadRecipe(recipe)
      setSelectedRecipe(recipe)
      setBuilderOpen(true)
    },
    [loadRecipe],
  )

  const handleNewRecipe = useCallback(() => {
    reset()
    setSelectedRecipe(null)
    setBuilderOpen(true)
  }, [reset])

  const handleBack = useCallback(() => {
    setBuilderOpen(false)
    setSelectedRecipe(null)
  }, [])

  const handleExecute = useCallback(() => {
    const config = getConfig() as { name: string; type: string; [key: string]: unknown }
    executeMutation.mutate({ recipe: config.name, overrides: config }, {
      onSuccess: (job) => {
        setSubmittedJobId(job.id)
      },
      onError: (err) => {
        toast.error(`Failed to submit job: ${err.message}`)
        setExecuteDialogOpen(false)
      },
    })
  }, [getConfig, executeMutation])

  const handleSave = useCallback(() => {
    const config = getConfig() as Record<string, unknown>
    const { name, type, description, ...rest } = config
    saveMutation.mutate(
      { name: name as string, type: type as string, description: (description as string) ?? "", config: rest },
      {
        onSuccess: () => toast.success(`Recipe "${name}" saved`),
        onError: (err) => toast.error(`Failed to save: ${err.message}`),
      },
    )
  }, [getConfig, saveMutation])

  const handleSaveAs = useCallback(
    (newName: string) => {
      setField("name", newName)
      const config = getConfig() as Record<string, unknown>
      const { type, description, ...rest } = config
      saveMutation.mutate(
        { name: newName, type: type as string, description: (description as string) ?? "", config: rest },
        {
          onSuccess: () => toast.success(`Recipe saved as "${newName}"`),
          onError: (err) => toast.error(`Failed to save: ${err.message}`),
        },
      )
    },
    [setField, getConfig, saveMutation],
  )

  if (isError) {
    return <ErrorState message={error?.message} onRetry={() => void refetch()} />
  }

  if (builderOpen) {
    const schema = selectedRecipe?.schema ?? null

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="sm" onClick={handleBack}>
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">
                {state.form.name || "New Recipe"}
              </h1>
              <p className="text-sm text-muted-foreground">
                {selectedRecipe
                  ? "Edit recipe configuration"
                  : "Create a new recipe"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <JsonEditorDialog
              jsonValue={state.json}
              jsonError={state.jsonError}
              schema={schema}
              onJsonChange={setJson}
              open={jsonEditorOpen}
              onOpenChange={setJsonEditorOpen}
            />

            <Button
              variant="outline"
              size="sm"
              onClick={handleSave}
              disabled={!state.form.name.trim()}
              data-testid="save-recipe"
            >
              <Save className="mr-2 h-4 w-4" />
              Save
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={() => setSaveDialogOpen(true)}
              data-testid="save-as-recipe"
            >
              <Copy className="mr-2 h-4 w-4" />
              Save As
            </Button>

            <Button
              size="sm"
              onClick={() => setExecuteDialogOpen(true)}
              disabled={!!state.jsonError}
              data-testid="execute-recipe"
            >
              <Play className="mr-2 h-4 w-4" />
              Execute
            </Button>
          </div>
        </div>

        <RecipeBuilderGuide type={state.form.type} hasDatasets={datasets.length > 0} hasModels={models.length > 0} />

        <RecipeBuilderForm
          form={state.form}
          onFieldChange={setField}
          datasets={datasets}
          models={models}
        />

        <ExecuteDialog
          open={executeDialogOpen}
          onOpenChange={(open) => {
            setExecuteDialogOpen(open)
            if (!open) setSubmittedJobId(null)
          }}
          config={getConfig()}
          onConfirm={handleExecute}
          isPending={executeMutation.isPending}
          submittedJobId={submittedJobId}
          onViewJob={() => {
            const jobId = submittedJobId
            setExecuteDialogOpen(false)
            setTimeout(() => {
              setSubmittedJobId(null)
              navigate(jobId ? `/jobs?job=${encodeURIComponent(jobId)}` : "/jobs")
            }, 200)
          }}
        />

        <SaveDialog
          open={saveDialogOpen}
          onOpenChange={setSaveDialogOpen}
          onSave={handleSaveAs}
          defaultName={state.form.name ? `${state.form.name} (copy)` : ""}
        />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div data-tutorial="recipe-browser" className="space-y-4">
        <PageHeader
          title="Recipes"
          actions={
            <Button onClick={handleNewRecipe} data-testid="new-recipe">
              New Recipe
            </Button>
          }
        />

        {/* How to build a model */}
        <div className="animate-welcome rounded-xl border bg-card p-5 space-y-5">
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Getting started</p>
          <h3 className="text-base font-semibold">Build a model in 5 steps</h3>
        </div>

        <div className="space-y-2.5">
          <div className="animate-message-in rounded-lg border border-transparent p-4 transition-all duration-300 hover:bg-muted/40 hover:border-border/60 hover:shadow-sm" data-tutorial="recipe-step-1">
            <div className="flex items-start gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background text-xs font-semibold">1</div>
              <div>
                <h4 className="text-sm font-medium">Set up a compute backend</h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Go to <Link to="/settings" className="underline underline-offset-2 hover:text-foreground transition-colors">Settings</Link> and
                  ensure a compute backend is configured (local GPU, SSH remote, or Kubernetes). Training requires at least one GPU.
                </p>
              </div>
            </div>
          </div>

          <div
            className="animate-message-in rounded-lg border border-transparent p-4 transition-all duration-300 hover:bg-muted/40 hover:border-border/60 hover:shadow-sm"
            style={{ animationDelay: "80ms" }}
            data-tutorial="recipe-step-2"
          >
            <div className="flex items-start gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background text-xs font-semibold">2</div>
              <div>
                <h4 className="text-sm font-medium">Generate training data with SDG</h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Pick an <span className="font-medium text-[#5e40be] dark:text-[#876fd4]">SDG</span> recipe below (e.g. <em>classification</em>, <em>extraction</em>, or <em>conversation</em>).
                  Configure the teacher model and sample count, then click Execute. This creates a labeled dataset in MLflow.
                  Use a local model (e.g. Granite via vLLM) or a cloud provider via <Link to="/settings" className="underline underline-offset-2 hover:text-foreground transition-colors">Settings → AI Gateway</Link>.
                </p>
              </div>
            </div>
          </div>

          <div
            className="animate-message-in rounded-lg border border-transparent p-4 transition-all duration-300 hover:bg-muted/40 hover:border-border/60 hover:shadow-sm"
            style={{ animationDelay: "160ms" }}
            data-tutorial="recipe-step-3"
          >
            <div className="flex items-start gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background text-xs font-semibold">3</div>
              <div>
                <h4 className="text-sm font-medium">Fine-tune a model</h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Pick a <span className="font-medium text-[#0066cc] dark:text-[#4394e5]">Training</span> recipe (e.g. <em>lora-sft</em> or a model-specific template like <em>Qwen3 4B LoRA</em>).
                  Select your generated dataset, choose a base model, and adjust hyperparameters (learning rate, epochs, batch size). Click Execute to start training.
                </p>
              </div>
            </div>
          </div>

          <div
            className="animate-message-in rounded-lg border border-transparent p-4 transition-all duration-300 hover:bg-muted/40 hover:border-border/60 hover:shadow-sm"
            style={{ animationDelay: "240ms" }}
            data-tutorial="recipe-step-4"
          >
            <div className="flex items-start gap-3">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-foreground text-background text-xs font-semibold">4</div>
              <div>
                <h4 className="text-sm font-medium">Monitor and iterate</h4>
                <p className="text-xs text-muted-foreground mt-1">
                  Track all jobs on the <Link to="/jobs" className="underline underline-offset-2 hover:text-foreground transition-colors">Jobs</Link> page.
                  View your datasets in <Link to="/datasets" className="underline underline-offset-2 hover:text-foreground transition-colors">Datasets</Link> and
                  trained models in <Link to="/models" className="underline underline-offset-2 hover:text-foreground transition-colors">Models</Link>.
                  If accuracy isn't high enough, generate more data or try a larger base model.
                </p>
              </div>
            </div>
          </div>
        </div>

        <p className="text-xs text-muted-foreground pt-3 border-t">
          Prefer a guided experience?{" "}
          <Link to="/chat" className="inline-flex items-center gap-1 underline underline-offset-2 hover:text-foreground transition-colors">
            Chat with Morty <ArrowRight className="h-3 w-3" />
          </Link>{" "}
          — describe what you want to build and he'll handle these steps for you.
        </p>
      </div>
      </div>

      {/* Category cards */}
      <div className="animate-welcome grid gap-4 md:grid-cols-2">
        <button
          onClick={() => { setTypeFilter("sdg"); setPage(0) }}
          className={`group rounded-xl border bg-card p-5 text-left transition-all duration-300 hover:border-[#b6a6e9] hover:shadow-md hover:-translate-y-0.5 dark:hover:border-[#21134d] ${typeFilter === "sdg" ? "border-[#b6a6e9] shadow-sm" : ""}`}
        >
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#ece6ff] text-[#5e40be] transition-colors group-hover:bg-[#d0c5f4] dark:bg-[#1b0d33]/40 dark:text-[#876fd4]">
              <Sparkles className="h-4.5 w-4.5" />
            </div>
            <div>
              <h3 className="font-semibold text-sm">Synthetic Data (SDG)</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Generate training data using teacher models</p>
              <p className="text-xs text-muted-foreground mt-1.5">{recipeCountByType.sdg} recipes</p>
            </div>
          </div>
        </button>

        <button
          onClick={() => { setTypeFilter("training"); setPage(0) }}
          className={`group rounded-xl border bg-card p-5 text-left transition-all duration-300 hover:border-[#92c5f9] hover:shadow-md hover:-translate-y-0.5 dark:hover:border-[#003366] ${typeFilter === "training" ? "border-[#92c5f9] shadow-sm" : ""}`}
        >
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[#e0f0ff] text-[#0066cc] transition-colors group-hover:bg-[#b9dafc] dark:bg-[#003366]/40 dark:text-[#4394e5]">
              <GraduationCap className="h-4.5 w-4.5" />
            </div>
            <div>
              <h3 className="font-semibold text-sm">Training</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Fine-tune models with SFT, LoRA, DPO, and more</p>
              <p className="text-xs text-muted-foreground mt-1.5">{recipeCountByType.training} recipes</p>
            </div>
          </div>
        </button>

      </div>

      {/* Filter tabs + search */}
      <div className="flex items-center justify-between gap-4">
        <Tabs value={typeFilter} onValueChange={(v) => { setTypeFilter(v); setPage(0) }}>
          <TabsList>
            <TabsTrigger value="all">All ({visibleRecipes.length})</TabsTrigger>
            <TabsTrigger value="sdg">SDG ({recipeCountByType.sdg})</TabsTrigger>
            <TabsTrigger value="training">Training ({recipeCountByType.training})</TabsTrigger>
          </TabsList>
        </Tabs>
        <SearchInput value={search} onChange={(v) => { setSearch(v); setPage(0) }} placeholder="Search recipes..." />
      </div>

      {isLoading ? (
        <TableSkeleton columns={4} />
      ) : (
        <RecipeTable
          recipes={filteredRecipes}
          page={page}
          onPageChange={setPage}
          onSelectRecipe={handleSelectRecipe}
          onCreateNew={handleNewRecipe}
        />
      )}
    </div>
  )
}
