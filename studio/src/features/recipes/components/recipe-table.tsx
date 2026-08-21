import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { BookOpen, Info, Copy, Check } from "lucide-react"
import { useState } from "react"
import { EmptyState } from "@/components/empty-state"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationPrevious,
  PaginationNext,
} from "@/components/ui/pagination"
import {
  formatRecipeType,
  recipeTypeClassName,
} from "../lib/format"
import type { RecipeEntry } from "../lib/format"
import { formatDate } from "@/lib/utils"
import type { Job } from "@/types/api"

const PAGE_SIZE = 20

interface RecipeTableProps {
  recipes: RecipeEntry[]
  page: number
  onPageChange: (page: number) => void
  onSelectJob: (job: Job, name: string) => void
}

export function RecipeTable({
  recipes,
  page,
  onPageChange,
  onSelectJob,
}: RecipeTableProps) {
  const totalPages = Math.max(1, Math.ceil(recipes.length / PAGE_SIZE))
  const start = page * PAGE_SIZE
  const pageItems = recipes.slice(start, start + PAGE_SIZE)

  return (
    <div>
      <Table className="table-fixed">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[45%]">Recipe</TableHead>
            <TableHead className="w-[12%]">Type</TableHead>
            <TableHead className="w-[15%]">ID</TableHead>
            <TableHead className="w-[28%]">Created</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {pageItems.length === 0 ? (
            <TableRow>
              <TableCell colSpan={4} className="p-0">
                <EmptyState
                  icon={BookOpen}
                  title="No recipes yet"
                  description="Run an SDG or training job to see its config here as a reusable recipe."
                />
              </TableCell>
            </TableRow>
          ) : (
            pageItems.map((entry) => (
              <TableRow
                key={entry.job.id}
                className="cursor-pointer"
                onClick={() => onSelectJob(entry.job, entry.name)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    onSelectJob(entry.job, entry.name)
                  }
                }}
              >
                <TableCell>
                  <div className="flex items-center gap-1.5 min-w-0">
                    <p className="font-medium text-sm truncate">{entry.name}</p>
                    {entry.version && (
                      <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-muted text-muted-foreground">
                        {entry.version}
                      </span>
                    )}
                  </div>
                </TableCell>
                <TableCell>
                  <Badge className={recipeTypeClassName(entry.job.type)}>
                    {formatRecipeType(entry.job.type)}
                  </Badge>
                </TableCell>
                <TableCell>
                  <CopyableId id={entry.job.id} />
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {formatDate(entry.job.created_at, { includeTime: true })}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      {totalPages > 1 && (
        <Pagination className="mt-4">
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious
                onClick={() => onPageChange(page - 1)}
                className={page === 0 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                aria-disabled={page === 0}
              />
            </PaginationItem>
            <PaginationItem>
              <span className="text-sm text-muted-foreground px-2">
                Page {page + 1} of {totalPages}
              </span>
            </PaginationItem>
            <PaginationItem>
              <PaginationNext
                onClick={() => onPageChange(page + 1)}
                className={page >= totalPages - 1 ? "pointer-events-none opacity-50" : "cursor-pointer"}
                aria-disabled={page >= totalPages - 1}
              />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      )}
    </div>
  )
}

function CopyableId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false)
  const short = id.length > 8 ? `${id.slice(0, 8)}…` : id

  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation()
    void navigator.clipboard.writeText(id)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="flex items-center gap-1">
      <button
        onClick={handleCopy}
        className="inline-flex items-center gap-1 font-mono text-xs text-muted-foreground hover:text-foreground transition-colors"
        title={id}
      >
        {short}
        {copied ? (
          <Check className="h-3 w-3 text-emerald-500" />
        ) : (
          <Copy className="h-3 w-3" />
        )}
      </button>
      <span title={`Earliest job with this config: ${id}`}>
        <Info className="h-3 w-3 text-muted-foreground/40 shrink-0" />
      </span>
    </div>
  )
}
