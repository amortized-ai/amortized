"use client";

import { useEffect, useState, useCallback, use } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  getJob,
  getJobMetrics,
  getJobArtifacts,
  cancelJob,
  type Job,
  type TrainingMetric,
  type Artifact,
} from "@/lib/api";

const statusVariant: Record<string, "success" | "info" | "warning" | "destructive" | "secondary"> = {
  completed: "success",
  running: "info",
  pending: "warning",
  failed: "destructive",
  cancelled: "secondary",
};

function formatBytes(bytes: number): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export default function JobDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetric[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const fetchJob = useCallback(() => {
    getJob(id).then(setJob).catch((e) => setError(e.message));
  }, [id]);

  const fetchMetrics = useCallback(() => {
    getJobMetrics(id).then(setMetrics).catch(() => {});
  }, [id]);

  const fetchArtifacts = useCallback(() => {
    getJobArtifacts(id).then(setArtifacts).catch(() => {});
  }, [id]);

  useEffect(() => {
    fetchJob();
    fetchMetrics();
    fetchArtifacts();
  }, [fetchJob, fetchMetrics, fetchArtifacts]);

  // Auto-refresh metrics for running jobs
  useEffect(() => {
    if (!job || job.status !== "running") return;
    const interval = setInterval(() => {
      fetchJob();
      fetchMetrics();
    }, 2000);
    return () => clearInterval(interval);
  }, [job?.status, fetchJob, fetchMetrics]);

  const handleCancel = async () => {
    setCancelling(true);
    try {
      const updated = await cancelJob(id);
      setJob(updated);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Cancel failed");
    } finally {
      setCancelling(false);
    }
  };

  if (error && !job) {
    return (
      <div>
        <Button variant="ghost" onClick={() => router.push("/jobs")} className="mb-4">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Jobs
        </Button>
        <p className="text-red-400">Error: {error}</p>
      </div>
    );
  }

  if (!job) {
    return <p className="text-muted-foreground">Loading...</p>;
  }

  const configEntries = Object.entries(job.config).filter(
    ([, v]) => v !== null && v !== undefined
  );

  return (
    <div>
      <Button variant="ghost" onClick={() => router.push("/jobs")} className="mb-4">
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Jobs
      </Button>

      <div className="flex items-center gap-3 mb-6">
        <h1 className="text-2xl font-bold font-mono">{job.id.slice(0, 8)}</h1>
        <Badge variant={job.type === "training" ? "default" : "secondary"}>
          {job.type}
        </Badge>
        <Badge variant={statusVariant[job.status] ?? "secondary"}>
          {job.status}
        </Badge>
        {(job.status === "pending" || job.status === "running") && (
          <Button
            variant="destructive"
            size="sm"
            onClick={handleCancel}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling..." : "Cancel"}
          </Button>
        )}
      </div>

      <div className="flex gap-4 text-sm text-muted-foreground mb-6">
        <span>Created: {new Date(job.created_at).toLocaleString()}</span>
        {job.started_at && (
          <span>Started: {new Date(job.started_at).toLocaleString()}</span>
        )}
        {job.completed_at && (
          <span>Finished: {new Date(job.completed_at).toLocaleString()}</span>
        )}
      </div>

      <Tabs defaultValue="overview">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          {job.error && (
            <Card className="mb-4 border-red-600/30">
              <CardHeader>
                <CardTitle className="text-red-400 text-sm">Error</CardTitle>
              </CardHeader>
              <CardContent>
                <pre className="text-sm text-red-300 whitespace-pre-wrap">{job.error}</pre>
              </CardContent>
            </Card>
          )}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm">
                {configEntries.map(([key, value]) => (
                  <div key={key} className="flex gap-2">
                    <dt className="text-muted-foreground font-mono">{key}:</dt>
                    <dd className="text-foreground">
                      {typeof value === "object" ? JSON.stringify(value) : String(value)}
                    </dd>
                  </div>
                ))}
              </dl>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics">
          {job.type !== "training" ? (
            <p className="text-muted-foreground">Metrics are only available for training jobs.</p>
          ) : metrics.length === 0 ? (
            <p className="text-muted-foreground">
              {job.status === "pending"
                ? "Waiting for training to start..."
                : "No metrics data available yet."}
            </p>
          ) : (
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Loss vs Step</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                      <XAxis
                        dataKey="step"
                        stroke="#737373"
                        fontSize={12}
                      />
                      <YAxis stroke="#737373" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1a1a1a",
                          border: "1px solid #262626",
                          borderRadius: "6px",
                          color: "#e5e5e5",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Learning Rate vs Step</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics.filter((m) => m.learning_rate != null)}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                      <XAxis
                        dataKey="step"
                        stroke="#737373"
                        fontSize={12}
                      />
                      <YAxis stroke="#737373" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1a1a1a",
                          border: "1px solid #262626",
                          borderRadius: "6px",
                          color: "#e5e5e5",
                        }}
                      />
                      <Line
                        type="monotone"
                        dataKey="learning_rate"
                        stroke="#10b981"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="artifacts">
          {artifacts.length === 0 ? (
            <p className="text-muted-foreground">No artifacts yet.</p>
          ) : (
            <div className="rounded-lg border border-border overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-muted/50">
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Type</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Path</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Size</th>
                    <th className="text-left px-4 py-3 font-medium text-muted-foreground">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {artifacts.map((artifact) => (
                    <tr key={artifact.id} className="border-b border-border">
                      <td className="px-4 py-3">
                        <Badge variant="secondary">{artifact.artifact_type}</Badge>
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-muted-foreground">
                        {artifact.path}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {formatBytes(artifact.size)}
                      </td>
                      <td className="px-4 py-3">
                        <Button variant="ghost" size="sm">
                          Download
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </TabsContent>

        <TabsContent value="logs">
          <Card>
            <CardContent className="pt-6">
              <p className="text-muted-foreground text-sm">
                Log viewer coming soon. Logs will be streamed in real-time for running jobs.
              </p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
