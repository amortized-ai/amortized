import { cn } from "@/lib/utils"

function Skeleton({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <rh-skeleton
      className={cn("block", className)}
      {...(props as React.HTMLAttributes<HTMLElement>)}
    />
  )
}

export { Skeleton }
