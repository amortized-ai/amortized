"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Select } from "@/components/ui/select";
import { listJobs, type Job, type JobType, type JobStatus } from "@/lib/api";

const statusVariant: Record<string, "success" | "info" | "warning" | "destructive" | "secondary"> = {
  completed: "success",
  running: "info",
  pending: "warning",
  failed: "destructive",
  cancelled: "secondary",
};

function formatDuration(job: Job): string {
  const start = job.started_at ? new Date(job.started_at) : null;
  const end = job.completed_at ? new Date(job.completed_at) : start ? new Date() : null;
  if (!start || !end) return "-";
  const seconds = Math.floor((end.getTime() - start.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function JobsPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [typeFilter, setTypeFilter] = useState<JobType | "">("");
  const [statusFilter, setStatusFilter] = useState<JobStatus | "">("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listJobs(statusFilter || undefined, typeFilter || undefined)
      .then((data) => {
        if (!cancelled) setJobs(data);
      })
      .catch(() => {
        if (!cancelled) setJobs([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [typeFilter, statusFilter]);

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Jobs</h1>
        <Button onClick={() => router.push("/jobs/new")}>
          <Plus className="h-4 w-4 mr-2" />
          New Job
        </Button>
      </div>

      <div className="flex gap-3 mb-4">
        <div className="flex gap-1">
          {(["", "training", "sdg"] as const).map((t) => (
            <Button
              key={t}
              variant={typeFilter === t ? "default" : "outline"}
              size="sm"
              onClick={() => setTypeFilter(t)}
            >
              {t === "" ? "All" : t === "training" ? "Training" : "SDG"}
            </Button>
          ))}
        </div>
        <Select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as JobStatus | "")}
          className="w-40"
        >
          <option value="">All statuses</option>
          <option value="pending">Pending</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
        </Select>
      </div>

      <div className="rounded-lg border border-border overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border bg-muted/50">
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Name / ID</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Type</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Status</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Model</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Created</th>
              <th className="text-left px-4 py-3 font-medium text-muted-foreground">Duration</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                  Loading...
                </td>
              </tr>
            ) : jobs.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-muted-foreground">
                  No jobs found
                </td>
              </tr>
            ) : (
              jobs.map((job) => (
                <tr
                  key={job.id}
                  className="border-b border-border hover:bg-muted/30 cursor-pointer transition-colors"
                  onClick={() => router.push(`/jobs/${job.id}`)}
                >
                  <td className="px-4 py-3 font-mono text-xs">
                    {job.id.slice(0, 8)}
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={job.type === "training" ? "default" : "secondary"}>
                      {job.type}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant={statusVariant[job.status] ?? "secondary"}>
                      {job.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {(job.config.model_path as string) ?? (job.config.model as string) ?? "-"}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {formatTime(job.created_at)}
                  </td>
                  <td className="px-4 py-3 text-muted-foreground">
                    {formatDuration(job)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
