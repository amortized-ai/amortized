import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { BookOpen } from "lucide-react"
import { EmptyState } from "@/components/empty-state"
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationPrevious,
  PaginationNext,
} from "@/components/ui/pagination"
import { formatRecipeName, formatRecipeType, recipeTypeClassName, getEffectiveType } from "../lib/format"
import type { Recipe } from "@/types/api"

const PAGE_SIZE = 25

interface RecipeTableProps {
  recipes: Recipe[]
  page: number
  onPageChange: (page: number) => void
  onSelectRecipe: (recipe: Recipe) => void
  onCreateNew?: () => void
}

export function RecipeTable({
  recipes,
  page,
  onPageChange,
  onSelectRecipe,
  onCreateNew,
}: RecipeTableProps) {
  const totalPages = Math.max(1, Math.ceil(recipes.length / PAGE_SIZE))
  const start = page * PAGE_SIZE
  const pageItems = recipes.slice(start, start + PAGE_SIZE)

  return (
    <div>
      <Table className="table-fixed">
        <TableHeader>
          <TableRow>
            <TableHead className="w-[30%]">Name</TableHead>
            <TableHead className="w-[10%]">Type</TableHead>
            <TableHead className="w-[50%]">Description</TableHead>
            <TableHead className="w-[10%]">Version</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {pageItems.length === 0 ? (
            <TableRow>
              <TableCell colSpan={4} className="p-0">
                <EmptyState
                  icon={BookOpen}
                  title="No recipes yet"
                  description="Create a recipe to configure training or SDG jobs."
                  actionLabel="Create Recipe"
                  onAction={onCreateNew}
                />
              </TableCell>
            </TableRow>
          ) : (
            pageItems.map((recipe) => (
              <TableRow
                key={recipe.name}
                className="cursor-pointer"
                onClick={() => onSelectRecipe(recipe)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault()
                    onSelectRecipe(recipe)
                  }
                }}
                data-testid={`recipe-row-${recipe.name}`}
              >
                <TableCell className="font-medium">{formatRecipeName(recipe.name)}</TableCell>
                <TableCell>
                  {(() => {
                    const effectiveType = getEffectiveType(recipe)
                    return effectiveType ? (
                      <Badge className={recipeTypeClassName(effectiveType)}>
                        {formatRecipeType(effectiveType)}
                      </Badge>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )
                  })()}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground max-w-[300px] truncate">
                  {recipe.description || "—"}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {recipe.version}
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
