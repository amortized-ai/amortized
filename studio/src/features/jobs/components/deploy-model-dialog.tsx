import { useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Rocket } from "lucide-react"
import { createServeJob, getJobs, getGpuAvailability } from "@/lib/api-client"
import type { Job } from "@/types/api"

interface DeployModelDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  /** Succeeded training jobs to offer as deploy sources. */
  trainingJobs: Job[]
}

/**
 * Manual model deployment: starts a serve job (persistent vLLM endpoint)
 * from a trained model or a public HF model. The chat (Morty) flow is the
 * other way to do this — same job type under the hood.
 */
export function DeployModelDialog({ open, onOpenChange, trainingJobs }: DeployModelDialogProps) {
  const [source, setSource] = useState<string>(trainingJobs.length > 0 ? "training" : "public")
  const [trainingJobId, setTrainingJobId] = useState<string>("")
  const [modelName, setModelName] = useState("")
  const [servedName, setServedName] = useState("")
  const [gpus, setGpus] = useState(1)
  const [maxModelLen, setMaxModelLen] = useState("")
  const [gpuMemUtil, setGpuMemUtil] = useState("0.9")

  const queryClient = useQueryClient()

  const { data: gpu } = useQuery({
    queryKey: ["gpu-availability"],
    queryFn: getGpuAvailability,
    enabled: open,
    refetchInterval: open ? 15000 : false,
  })

  const deploy = useMutation({
    mutationFn: (config: Record<string, unknown>) => createServeJob(config),
    onSuccess: (job) => {
      toast.success(`Serve job ${job.id.slice(0, 8)} submitted — deploying endpoint`)
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
      onOpenChange(false)
      resetForm()
    },
    onError: (err) => {
      toast.error(
        `Failed to deploy: ${err instanceof Error ? err.message : "Unknown error"}`,
      )
    },
  })

  function resetForm() {
    setTrainingJobId("")
    setModelName("")
    setServedName("")
    setGpus(1)
    setMaxModelLen("")
    setGpuMemUtil("0.9")
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const config: Record<string, unknown> = {
      nproc_per_node: gpus,
    }
    if (servedName.trim()) config.served_model_name = servedName.trim()
    const len = parseInt(maxModelLen, 10)
    if (maxModelLen.trim() && Number.isFinite(len) && len > 0) config.max_model_len = len
    const util = parseFloat(gpuMemUtil)
    if (Number.isFinite(util) && util > 0 && util <= 1) {
      config.vllm_args = [`--gpu-memory-utilization=${util}`]
    }
    if (source === "training") {
      if (!trainingJobId) return
      config.training_job_id = trainingJobId
    } else {
      if (!modelName.trim()) return
      config.model_name_or_path = modelName.trim()
    }
    deploy.mutate(config)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-teal-100 dark:bg-teal-900/30">
              <Rocket className="h-5 w-5 text-teal-700 dark:text-teal-300" />
            </div>
            <div className="flex-1">
              <DialogTitle className="text-lg font-semibold">Deploy model</DialogTitle>
              <DialogDescription className="mt-1 text-sm leading-relaxed">
                Start a persistent vLLM endpoint. It keeps its GPUs until you
                stop it from the Jobs page.
              </DialogDescription>
            </div>
          </div>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label>Model source</Label>
            <Select value={source} onValueChange={setSource}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {trainingJobs.length > 0 && (
                  <SelectItem value="training">A trained model</SelectItem>
                )}
                <SelectItem value="public">Public HF model</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {source === "training" ? (
            <div className="space-y-2">
              <Label>Trained model</Label>
              <Select value={trainingJobId} onValueChange={setTrainingJobId}>
                <SelectTrigger>
                  <SelectValue placeholder="Pick a training job" />
                </SelectTrigger>
                <SelectContent>
                  {trainingJobs.map((job) => {
                    const registered = job.config?.model_display_name
                    const display =
                      (typeof registered === "string" && registered) ||
                      (job.metadata?.name as string | undefined) ||
                      job.id.slice(0, 8)
                    return (
                      <SelectItem key={job.id} value={job.id}>
                        {display} (job {job.id.slice(0, 8)})
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="deploy-model-name">Model</Label>
              <Input
                id="deploy-model-name"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g. Qwen/Qwen3.5-2B"
                required
              />
            </div>
          )}

          <div className="space-y-2">
            <Label htmlFor="deploy-served-name">
              Served name{" "}
              <span className="font-normal text-muted-foreground">
                (optional — what clients pass as "model")
              </span>
            </Label>
            <Input
              id="deploy-served-name"
              value={servedName}
              onChange={(e) => setServedName(e.target.value)}
              placeholder={
                source === "training" ? "defaults to the trained model's name" : modelName || "defaults to the model name"
              }
            />
          </div>

          {gpu && (
            <div className="rounded-lg border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
              {gpu.available !== null ? (
                <>
                  <span className="font-medium text-foreground">
                    {gpu.available} GPU{gpu.available === 1 ? "" : "s"} available
                  </span>{" "}
                  within your budget
                  {gpu.quota_limit !== null && ` (${gpu.quota_used}/${gpu.quota_limit} in use)`}
                  {" · "}~{gpu.per_gpu_memory_gb} GB each
                </>
              ) : (
                <>~{gpu.per_gpu_memory_gb} GB per GPU · budget unavailable</>
              )}
            </div>
          )}

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-2">
              <Label htmlFor="deploy-gpus">GPUs</Label>
              <Input
                id="deploy-gpus"
                type="number"
                min={1}
                max={8}
                value={gpus}
                onChange={(e) => setGpus(parseInt(e.target.value, 10) || 1)}
              />
              {gpu != null &&
                gpu.available !== null &&
                gpu.available !== undefined &&
                gpus > gpu.available && (
                  <p className="text-xs text-rh-danger">
                    Exceeds your available budget ({gpu.available}).
                  </p>
                )}
            </div>
            <div className="space-y-2">
              <Label htmlFor="deploy-max-len">
                Max context{" "}
                <span className="font-normal text-muted-foreground">(optional)</span>
              </Label>
              <Input
                id="deploy-max-len"
                type="number"
                value={maxModelLen}
                onChange={(e) => setMaxModelLen(e.target.value)}
                placeholder="model default"
              />
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="deploy-gpu-mem">
              GPU memory utilization{" "}
              <span className="font-normal text-muted-foreground">
                (0-1, fraction of each GPU vLLM may use)
              </span>
            </Label>
            <Input
              id="deploy-gpu-mem"
              type="number"
              step="0.05"
              min={0.1}
              max={1}
              value={gpuMemUtil}
              onChange={(e) => setGpuMemUtil(e.target.value)}
              placeholder="0.9"
            />
          </div>

          <p className="text-xs text-muted-foreground">
            The deployment pre-checks that the model weights fit the available
            GPU memory and fails fast if not.
          </p>

          <DialogFooter className="gap-2 sm:gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            <Button type="submit" disabled={deploy.isPending} className="gap-2">
              <Rocket className="h-4 w-4" />
              {deploy.isPending ? "Submitting..." : "Deploy"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

/** Succeeded training jobs offered as deploy sources. */
export function useDeployableTrainingJobs(enabled: boolean) {
  return useQuery<Job[]>({
    queryKey: ["jobs", { type: "training", status: "succeeded" }],
    queryFn: () => getJobs({ type: "training", status: "succeeded" }, { per_page: 50 }),
    enabled,
  })
}
