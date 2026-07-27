import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  getPaginationRowModel,
  getFilteredRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
  type ColumnFiltersState,
  type PaginationState,
  type OnChangeFn,
} from "@tanstack/react-table"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ChevronLeft, ChevronRight, ArrowUpDown, ArrowUp, ArrowDown, Search } from "lucide-react"
import { useState, type ReactNode } from "react"

interface DataTableProps<TData> {
  columns: ColumnDef<TData, unknown>[]
  data: TData[]
  onRowClick?: (row: TData) => void
  emptyState?: ReactNode
  searchPlaceholder?: string
  searchColumn?: string
  pageSize?: number
  manualPagination?: boolean
  pageCount?: number
  pagination?: PaginationState
  onPaginationChange?: OnChangeFn<PaginationState>
  sorting?: SortingState
  onSortingChange?: OnChangeFn<SortingState>
  rowTestId?: (row: TData) => string
}

export function DataTable<TData>({
  columns,
  data,
  onRowClick,
  emptyState,
  searchPlaceholder = "Search...",
  searchColumn,
  pageSize = 10,
  manualPagination = false,
  pageCount,
  pagination: controlledPagination,
  onPaginationChange,
  sorting: controlledSorting,
  onSortingChange,
  rowTestId,
}: DataTableProps<TData>) {
  const [internalSorting, setInternalSorting] = useState<SortingState>([])
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([])
  const [internalPagination, setInternalPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize,
  })

  const sorting = controlledSorting ?? internalSorting
  const onSortingChangeFn = onSortingChange ?? setInternalSorting
  const pagination = controlledPagination ?? internalPagination
  const onPaginationChangeFn = onPaginationChange ?? setInternalPagination

  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data,
    columns,
    state: { sorting, columnFilters, pagination },
    onSortingChange: onSortingChangeFn,
    onColumnFiltersChange: setColumnFilters,
    onPaginationChange: onPaginationChangeFn,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: manualPagination ? undefined : getSortedRowModel(),
    getPaginationRowModel: manualPagination ? undefined : getPaginationRowModel(),
    getFilteredRowModel: manualPagination ? undefined : getFilteredRowModel(),
    manualPagination,
    ...(pageCount != null ? { pageCount } : {}),
  })

  const searchValue = searchColumn
    ? (table.getColumn(searchColumn)?.getFilterValue() as string) ?? ""
    : ""

  return (
    <div className="space-y-2">
      {searchColumn && (
        <div className="relative max-w-sm">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder={searchPlaceholder}
            value={searchValue}
            onChange={(e) =>
              table.getColumn(searchColumn)?.setFilterValue(e.target.value)
            }
            className="pl-8"
          />
        </div>
      )}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead key={header.id}>
                    {header.isPlaceholder ? null : header.column.getCanSort() ? (
                      <button
                        className="flex items-center gap-1 hover:text-foreground"
                        onClick={header.column.getToggleSortingHandler()}
                      >
                        {flexRender(header.column.columnDef.header, header.getContext())}
                        {header.column.getIsSorted() === "asc" ? (
                          <ArrowUp className="h-3 w-3" />
                        ) : header.column.getIsSorted() === "desc" ? (
                          <ArrowDown className="h-3 w-3" />
                        ) : (
                          <ArrowUpDown className="h-3 w-3 opacity-50" />
                        )}
                      </button>
                    ) : (
                      flexRender(header.column.columnDef.header, header.getContext())
                    )}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="p-0">
                  {emptyState ?? (
                    <div className="py-8 text-center text-sm text-muted-foreground">
                      No data
                    </div>
                  )}
                </TableCell>
              </TableRow>
            ) : (
              table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  className={onRowClick ? "cursor-pointer" : ""}
                  onClick={() => onRowClick?.(row.original)}
                  role={onRowClick ? "button" : undefined}
                  tabIndex={onRowClick ? 0 : undefined}
                  onKeyDown={
                    onRowClick
                      ? (e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault()
                            onRowClick(row.original)
                          }
                        }
                      : undefined
                  }
                  data-testid={rowTestId?.(row.original)}
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(cell.column.columnDef.cell, cell.getContext())}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {table.getPageCount() > 1 && (
        <div className="flex items-center justify-between px-2">
          <span className="text-sm text-muted-foreground">
            Page {table.getState().pagination.pageIndex + 1} of{" "}
            {table.getPageCount()}
          </span>
          <div className="flex gap-1">
            <Button
              variant="outline"
              size="sm"
              disabled={!table.getCanPreviousPage()}
              onClick={() => table.previousPage()}
              aria-label="Previous page"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={!table.getCanNextPage()}
              onClick={() => table.nextPage()}
              aria-label="Next page"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}