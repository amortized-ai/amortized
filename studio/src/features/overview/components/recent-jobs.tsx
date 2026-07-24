import { Link, useNavigate } from "react-router"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Empty,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
  EmptyDescription,
  EmptyContent,
} from "@/components/ui/empty"
import { ArrowRight, Briefcase } from "lucide-react"
import { StatusBadge } from "@/features/jobs/components/status-badge"
import { JobTypeBadge } from "@/features/jobs/components/job-type-badge"
import { useJobs } from "@/features/jobs/api/use-jobs"
import { formatDate } from "@/lib/utils"

export function RecentJobs() {
  const navigate = useNavigate()
  const { data: jobs = [], isLoading } = useJobs()
  const recentJobs = jobs.slice(0, 5)

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">Recent Jobs</CardTitle>
        <Button variant="ghost" size="sm" asChild>
          <Link to="/jobs">
            View all
            <ArrowRight className="ml-1 h-3 w-3" />
          </Link>
        </Button>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-10 w-full" />
            ))}
          </div>
        ) : recentJobs.length === 0 ? (
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Briefcase />
              </EmptyMedia>
              <EmptyTitle>No jobs yet</EmptyTitle>
              <EmptyDescription>
                Create a recipe to get started.
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button size="sm" asChild>
                <Link to="/recipes">Create a Recipe</Link>
              </Button>
            </EmptyContent>
          </Empty>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Name</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Created</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recentJobs.map((job) => (
                <TableRow
                  key={job.id}
                  className="cursor-pointer hover:bg-muted/50"
                  onClick={() => navigate(`/jobs?job=${encodeURIComponent(job.id)}`)}
                >
                  <TableCell className="font-medium text-sm">
                    {(job.metadata?.name as string) || job.id.slice(0, 8)}
                  </TableCell>
                  <TableCell>
                    <JobTypeBadge type={job.type} />
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={job.status} />
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(job.created_at)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}
