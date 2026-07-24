import { useCallback, useMemo } from "react"
import { useSearchParams } from "react-router"
import type { JobType, JobStatus } from "@/types/api"

export function useJobFilters() {
  const [searchParams, setSearchParams] = useSearchParams()

  const typeFilter = useMemo<JobType[]>(() => {
    const param = searchParams.get("type")
    return param ? (param.split(",") as JobType[]) : []
  }, [searchParams])

  const statusFilter = useMemo<JobStatus[]>(() => {
    const param = searchParams.get("status")
    return param ? (param.split(",") as JobStatus[]) : []
  }, [searchParams])

  const setTypeFilter = useCallback(
    (types: JobType[]) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        if (types.length > 0) {
          next.set("type", types.join(","))
        } else {
          next.delete("type")
        }
        return next
      })
    },
    [setSearchParams],
  )

  const setStatusFilter = useCallback(
    (statuses: JobStatus[]) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev)
        if (statuses.length > 0) {
          next.set("status", statuses.join(","))
        } else {
          next.delete("status")
        }
        return next
      })
    },
    [setSearchParams],
  )

  return { typeFilter, statusFilter, setTypeFilter, setStatusFilter }
}
