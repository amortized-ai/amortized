"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  createTrainingJob,
  createSDGJob,
  estimateMemory,
  listFlows,
  type SDGFlow,
  type MemoryEstimate,
} from "@/lib/api";

export default function NewJobPage() {
  const router = useRouter();
  const [tab, setTab] = useState("training");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Training form state
  const [modelPath, setModelPath] = useState("");
  const [dataPath, setDataPath] = useState("");
  const [ckptOutputDir, setCkptOutputDir] = useState("");
  const [learningRate, setLearningRate] = useState("");
  const [numEpochs, setNumEpochs] = useState("");
  const [loraR, setLoraR] = useState("");
  const [loraAlpha, setLoraAlpha] = useState("");
  const [loadIn4bit, setLoadIn4bit] = useState(false);
  const [microBatchSize, setMicroBatchSize] = useState("");
  const [maxSeqLen, setMaxSeqLen] = useState("");

  // VRAM estimate
  const [vramEstimate, setVramEstimate] = useState<MemoryEstimate | null>(null);
  const [estimating, setEstimating] = useState(false);

  // SDG form state
  const [flows, setFlows] = useState<SDGFlow[]>([]);
  const [flowId, setFlowId] = useState("");
  const [datasetPath, setDatasetPath] = useState("");
  const [teacherModel, setTeacherModel] = useState("");
  const [apiBase, setApiBase] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [runtimeParams, setRuntimeParams] = useState("");

  useEffect(() => {
    listFlows().then(setFlows).catch(() => {});
  }, []);

  const handleEstimateVRAM = async () => {
    if (!modelPath) return;
    setEstimating(true);
    setVramEstimate(null);
    try {
      const est = await estimateMemory({
        model_path: modelPath,
        lora_r: loraR ? parseInt(loraR) : undefined,
        batch_size: microBatchSize ? parseInt(microBatchSize) : undefined,
        max_seq_len: maxSeqLen ? parseInt(maxSeqLen) : undefined,
        load_in_4bit: loadIn4bit,
      });
      setVramEstimate(est);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Estimation failed");
    } finally {
      setEstimating(false);
    }
  };

  const handleSubmitTraining = async () => {
    if (!modelPath || !dataPath || !ckptOutputDir) {
      setError("model_path, data_path, and ckpt_output_dir are required");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const job = await createTrainingJob({
        model_path: modelPath,
        data_path: dataPath,
        ckpt_output_dir: ckptOutputDir,
        learning_rate: learningRate ? parseFloat(learningRate) : null,
        num_epochs: numEpochs ? parseInt(numEpochs) : null,
        lora_r: loraR ? parseInt(loraR) : null,
        lora_alpha: loraAlpha ? parseInt(loraAlpha) : null,
        load_in_4bit: loadIn4bit || null,
        micro_batch_size: microBatchSize ? parseInt(microBatchSize) : null,
        max_seq_len: maxSeqLen ? parseInt(maxSeqLen) : null,
      });
      router.push(`/jobs/${job.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create job");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitSDG = async () => {
    if (!flowId || !datasetPath || !teacherModel) {
      setError("flow_id, dataset_path, and model are required");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      let parsedParams: Record<string, unknown> | null = null;
      if (runtimeParams.trim()) {
        try {
          parsedParams = JSON.parse(runtimeParams);
        } catch {
          setError("runtime_params must be valid JSON");
          setSubmitting(false);
          return;
        }
      }
      const job = await createSDGJob({
        flow_id: flowId,
        dataset_path: datasetPath,
        model: teacherModel,
        api_base: apiBase || null,
        api_key: apiKey || null,
        runtime_params: parsedParams,
      });
      router.push(`/jobs/${job.id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create job");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="max-w-2xl">
      <h1 className="text-2xl font-bold mb-6">New Job</h1>

      {error && (
        <div className="bg-red-600/10 border border-red-600/30 rounded-md px-4 py-3 mb-4 text-sm text-red-400">
          {error}
        </div>
      )}

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList className="mb-4">
          <TabsTrigger value="training">Training</TabsTrigger>
          <TabsTrigger value="sdg">SDG</TabsTrigger>
        </TabsList>

        <TabsContent value="training">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">LoRA SFT Training Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="model_path">Model Path *</Label>
                <Input
                  id="model_path"
                  placeholder="Qwen/Qwen2.5-1.5B-Instruct"
                  value={modelPath}
                  onChange={(e) => setModelPath(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="data_path">Data Path *</Label>
                <Input
                  id="data_path"
                  placeholder="/data/training.jsonl"
                  value={dataPath}
                  onChange={(e) => setDataPath(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="ckpt_output_dir">Output Directory *</Label>
                <Input
                  id="ckpt_output_dir"
                  placeholder="/outputs/my-adapter"
                  value={ckptOutputDir}
                  onChange={(e) => setCkptOutputDir(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="learning_rate">Learning Rate</Label>
                  <Input
                    id="learning_rate"
                    type="number"
                    step="0.0001"
                    placeholder="2e-4"
                    value={learningRate}
                    onChange={(e) => setLearningRate(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="num_epochs">Epochs</Label>
                  <Input
                    id="num_epochs"
                    type="number"
                    min="1"
                    placeholder="3"
                    value={numEpochs}
                    onChange={(e) => setNumEpochs(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lora_r">LoRA Rank</Label>
                  <Input
                    id="lora_r"
                    type="number"
                    min="1"
                    placeholder="16"
                    value={loraR}
                    onChange={(e) => setLoraR(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="lora_alpha">LoRA Alpha</Label>
                  <Input
                    id="lora_alpha"
                    type="number"
                    min="1"
                    placeholder="32"
                    value={loraAlpha}
                    onChange={(e) => setLoraAlpha(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="micro_batch_size">Micro Batch Size</Label>
                  <Input
                    id="micro_batch_size"
                    type="number"
                    min="1"
                    placeholder="2"
                    value={microBatchSize}
                    onChange={(e) => setMicroBatchSize(e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="max_seq_len">Max Sequence Length</Label>
                  <Input
                    id="max_seq_len"
                    type="number"
                    min="1"
                    placeholder="2048"
                    value={maxSeqLen}
                    onChange={(e) => setMaxSeqLen(e.target.value)}
                  />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  id="load_in_4bit"
                  type="checkbox"
                  checked={loadIn4bit}
                  onChange={(e) => setLoadIn4bit(e.target.checked)}
                  className="rounded border-border"
                />
                <Label htmlFor="load_in_4bit">Enable QLoRA (4-bit quantization)</Label>
              </div>

              <div className="flex gap-3 pt-4">
                <Button
                  variant="outline"
                  onClick={handleEstimateVRAM}
                  disabled={!modelPath || estimating}
                >
                  {estimating ? "Estimating..." : "Estimate VRAM"}
                </Button>
                <Button onClick={handleSubmitTraining} disabled={submitting}>
                  {submitting ? "Submitting..." : "Submit Job"}
                </Button>
              </div>

              {vramEstimate && (
                <div className="bg-muted rounded-md px-4 py-3 text-sm mt-2">
                  <p className="font-medium mb-1">VRAM Estimate</p>
                  <p className="text-muted-foreground">
                    Estimated VRAM: <span className="text-foreground font-mono">{vramEstimate.estimated_vram_gb.toFixed(1)} GB</span>
                    {" | "}LoRA r={vramEstimate.lora_r}, batch={vramEstimate.batch_size}, seq_len={vramEstimate.max_seq_len}
                    {vramEstimate.load_in_4bit ? " (QLoRA)" : ""}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="sdg">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Synthetic Data Generation Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="flow_id">SDG Flow *</Label>
                <Select
                  id="flow_id"
                  value={flowId}
                  onChange={(e) => setFlowId(e.target.value)}
                >
                  <option value="">Select a flow...</option>
                  {flows.map((flow) => (
                    <option key={flow.id} value={flow.id}>
                      {flow.name}
                    </option>
                  ))}
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="dataset_path">Dataset Path *</Label>
                <Input
                  id="dataset_path"
                  placeholder="/data/seed-dataset.jsonl"
                  value={datasetPath}
                  onChange={(e) => setDatasetPath(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="teacher_model">Teacher Model *</Label>
                <Input
                  id="teacher_model"
                  placeholder="openai/gpt-4o"
                  value={teacherModel}
                  onChange={(e) => setTeacherModel(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="api_base">API Base URL</Label>
                <Input
                  id="api_base"
                  placeholder="https://api.openai.com/v1"
                  value={apiBase}
                  onChange={(e) => setApiBase(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="api_key">API Key</Label>
                <Input
                  id="api_key"
                  type="password"
                  placeholder="sk-..."
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="runtime_params">Runtime Parameters (JSON)</Label>
                <Textarea
                  id="runtime_params"
                  placeholder='{"temperature": 0.7}'
                  rows={3}
                  value={runtimeParams}
                  onChange={(e) => setRuntimeParams(e.target.value)}
                />
              </div>

              <div className="pt-4">
                <Button onClick={handleSubmitSDG} disabled={submitting}>
                  {submitting ? "Submitting..." : "Submit Job"}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
