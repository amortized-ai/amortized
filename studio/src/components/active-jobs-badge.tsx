import { Link } from "react-router"
import { useJobs } from "@/features/jobs/api/use-jobs"

export function ActiveJobsBadge() {
  const { data: jobs = [] } = useJobs({ status: "running" })
  const count = jobs.length

  if (count === 0) return null

  return (
    <Link
      to="/jobs?status=running"
      className="text-xs text-muted-foreground hover:text-foreground"
    >
      {count} {count === 1 ? "job" : "jobs"} running
    </Link>
  )
}
