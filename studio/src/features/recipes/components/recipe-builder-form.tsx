import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Label } from "@/components/ui/label"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Link } from "react-router"
import type { RecipeFormState, RecipeType, TrainingMethod } from "../hooks/use-recipe-state"
import type { DatasetRecord, ModelRecord } from "@/types/api"

interface RecipeBuilderFormProps {
  form: RecipeFormState
  onFieldChange: (field: keyof RecipeFormState, value: string) => void
  datasets: DatasetRecord[]
  models: ModelRecord[]
}

const TRAINING_METHODS: { value: TrainingMethod; label: string }[] = [
  { value: "lora_sft", label: "LoRA SFT" },
  { value: "sft", label: "SFT" },
  { value: "osft", label: "OSFT" },
  { value: "dpo", label: "DPO" },
  { value: "grpo", label: "GRPO" },
  { value: "lora_grpo", label: "LoRA GRPO" },
  { value: "kto", label: "KTO" },
  { value: "gepa", label: "GEPA" },
  { value: "gkd", label: "GKD" },
]

function FieldLabel({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return <Label htmlFor={htmlFor}>{children}</Label>
}

function TrainingSection({
  form,
  onFieldChange,
  datasets,
  models,
}: Pick<RecipeBuilderFormProps, "form" | "onFieldChange" | "datasets" | "models">) {
  return (
    <Accordion type="multiple" defaultValue={["method", "model", "data"]}>
      <AccordionItem value="method">
        <AccordionTrigger>Training Method</AccordionTrigger>
        <AccordionContent>
          <ToggleGroup
            type="single"
            variant="outline"
            value={form.training_method}
            onValueChange={(v) => { if (v) onFieldChange("training_method", v) }}
            className="flex flex-wrap justify-start gap-2"
            spacing={1}
          >
            {TRAINING_METHODS.map((m) => (
              <ToggleGroupItem
                key={m.value}
                value={m.value}
                data-testid={`method-${m.value}`}
                className="rounded-lg border border-border/60 data-[state=on]:border-primary/40 data-[state=on]:bg-rh-blue-light/50 data-[state=on]:text-rh-blue-dark dark:data-[state=on]:border-primary/50 dark:data-[state=on]:bg-rh-blue-dark/40 dark:data-[state=on]:text-primary transition-all duration-200 hover:border-border"
              >
                {m.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="model">
        <AccordionTrigger>Model Selection</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-2">
            <FieldLabel>Base Model</FieldLabel>
            <Select
              value={form.base_model}
              onValueChange={(v) => onFieldChange("base_model", v)}
            >
              <SelectTrigger data-testid="base-model-select">
                <SelectValue placeholder="Select base model" />
              </SelectTrigger>
              <SelectContent>
                {models.length === 0 ? (
                  <SelectItem value="__none" disabled>
                    No models available
                  </SelectItem>
                ) : (
                  models.map((m) => (
                    <SelectItem key={m.name} value={m.name}>
                      {m.name}
                    </SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="data">
        <AccordionTrigger>Data</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <FieldLabel>Training Dataset</FieldLabel>
              <Select
                value={form.training_dataset}
                onValueChange={(v) => onFieldChange("training_dataset", v)}
              >
                <SelectTrigger data-testid="training-dataset-select">
                  <SelectValue placeholder="Select training dataset" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.length === 0 ? (
                    <SelectItem value="__none" disabled>
                      No datasets available
                    </SelectItem>
                  ) : (
                    datasets.map((d) => (
                      <SelectItem key={d.run_id} value={d.run_id}>
                        {d.name}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <FieldLabel>Validation Dataset</FieldLabel>
              <Select
                value={form.validation_dataset}
                onValueChange={(v) => onFieldChange("validation_dataset", v)}
              >
                <SelectTrigger data-testid="validation-dataset-select">
                  <SelectValue placeholder="Select validation dataset (optional)" />
                </SelectTrigger>
                <SelectContent>
                  {datasets.map((d) => (
                    <SelectItem key={d.run_id} value={d.run_id}>
                      {d.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="advanced">
        <AccordionTrigger>Advanced Training Settings</AccordionTrigger>
        <AccordionContent>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <FieldLabel>Learning Rate</FieldLabel>
              <Input
                value={form.learning_rate}
                onChange={(e) => onFieldChange("learning_rate", e.target.value)}
                placeholder="2e-5"
                data-testid="learning-rate-input"
              />
            </div>
            <div className="space-y-2">
              <FieldLabel>Epochs</FieldLabel>
              <Input
                type="number"
                value={form.epochs}
                onChange={(e) => onFieldChange("epochs", e.target.value)}
                placeholder="3"
                data-testid="epochs-input"
              />
            </div>
            <div className="space-y-2">
              <FieldLabel>Batch Size</FieldLabel>
              <Input
                type="number"
                value={form.batch_size}
                onChange={(e) => onFieldChange("batch_size", e.target.value)}
                placeholder="8"
                data-testid="batch-size-input"
              />
            </div>
            <div className="space-y-2">
              <FieldLabel>LoRA Rank</FieldLabel>
              <Input
                type="number"
                value={form.lora_rank}
                onChange={(e) => onFieldChange("lora_rank", e.target.value)}
                placeholder="16"
                data-testid="lora-rank-input"
              />
            </div>
            <div className="space-y-2">
              <FieldLabel>LoRA Alpha</FieldLabel>
              <Input
                type="number"
                value={form.lora_alpha}
                onChange={(e) => onFieldChange("lora_alpha", e.target.value)}
                placeholder="32"
                data-testid="lora-alpha-input"
              />
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

function SdgSection({
  form,
  onFieldChange,
  datasets,
}: Pick<RecipeBuilderFormProps, "form" | "onFieldChange" | "datasets">) {
  return (
    <Accordion type="multiple" defaultValue={["teacher", "generation", "input"]}>
      <AccordionItem value="teacher">
        <AccordionTrigger>Teacher Model</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-2">
            <Input
              value={form.teacher_model}
              onChange={(e) => onFieldChange("teacher_model", e.target.value)}
              placeholder="e.g. granite-3.1-8b or openai/gpt-4o-mini"
              data-testid="teacher-model-input"
            />
            <p className="text-xs text-muted-foreground">
              <span className="font-medium">Local model:</span> use the model name (e.g. <span className="font-mono text-[11px]">granite-3.1-8b</span>) and set <span className="font-mono text-[11px]">api_base</span> in the JSON editor to your serving endpoint.
              <span className="font-medium ml-2">API provider:</span> use LiteLLM format <span className="font-mono text-[11px]">provider/model</span> with an API key.
            </p>
          </div>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="generation">
        <AccordionTrigger>Generation Settings</AccordionTrigger>
        <AccordionContent>
          <div className="space-y-4">
            <div className="space-y-2">
              <FieldLabel>Number of Samples</FieldLabel>
              <Input
                type="number"
                value={form.num_samples}
                onChange={(e) => onFieldChange("num_samples", e.target.value)}
                placeholder="100"
                data-testid="num-samples-input"
              />
            </div>
            <div className="space-y-2">
              <FieldLabel>Strategy Parameters (JSON)</FieldLabel>
              <Input
                value={form.strategy_params}
                onChange={(e) => onFieldChange("strategy_params", e.target.value)}
                placeholder="{}"
                data-testid="strategy-params-input"
              />
            </div>
          </div>
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="input">
        <AccordionTrigger>Input Data</AccordionTrigger>
        <AccordionContent>
          {datasets.length > 0 ? (
            <Select
              value={form.input_data}
              onValueChange={(v) => onFieldChange("input_data", v)}
            >
              <SelectTrigger data-testid="input-data-select">
                <SelectValue placeholder="Select input data" />
              </SelectTrigger>
              <SelectContent>
                {datasets.map((d) => (
                  <SelectItem key={d.run_id} value={d.run_id}>
                    {d.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : (
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">
                No datasets available.{" "}
                <Link to="/recipes" className="text-primary underline underline-offset-4 hover:text-primary/80">
                  Run an SDG recipe
                </Link>{" "}
                to generate training data.
              </p>
            </div>
          )}
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  )
}

export function RecipeBuilderForm(props: RecipeBuilderFormProps) {
  const { form, onFieldChange } = props

  return (
    <div className="mx-auto max-w-2xl space-y-8" data-testid="recipe-builder-form">
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <FieldLabel>Recipe Name</FieldLabel>
          <Input
            value={form.name}
            onChange={(e) => onFieldChange("name", e.target.value)}
            placeholder="My Recipe"
            data-testid="recipe-name-input"
          />
        </div>
        <div className="space-y-2">
          <FieldLabel>Type</FieldLabel>
          <Select
            value={form.type}
            onValueChange={(v) => onFieldChange("type", v as RecipeType)}
          >
            <SelectTrigger data-testid="recipe-type-select">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="training">Training</SelectItem>
              <SelectItem value="sdg">SDG</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="space-y-2">
        <FieldLabel>Description</FieldLabel>
        <Input
          value={form.description}
          onChange={(e) => onFieldChange("description", e.target.value)}
          placeholder="Describe this recipe..."
          data-testid="recipe-description-input"
        />
      </div>

      {form.type === "training" && (
        <TrainingSection
          form={form}
          onFieldChange={onFieldChange}
          datasets={props.datasets}
          models={props.models}
        />
      )}
      {form.type === "sdg" && (
        <SdgSection
          form={form}
          onFieldChange={onFieldChange}
          datasets={props.datasets}
        />
      )}
    </div>
  )
}
