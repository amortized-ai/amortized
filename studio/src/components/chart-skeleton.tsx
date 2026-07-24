import { Skeleton } from "@/components/ui/skeleton"

export function ChartSkeleton() {
  return (
    <div aria-busy="true" role="status">
      <Skeleton className="h-[300px] w-full rounded-lg" />
    </div>
  )
}
