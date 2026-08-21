import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { getJobs, createTrainingJob, createSdgJob, getConfigSchemas, getStarterTemplates } from "@/lib/api-client"
import type { Job, JobType } from "@/types/api"

export type JsonSchema = Record<string, unknown>

export function useConfigSchemas() {
  return useQuery<Record<string, JsonSchema>>({
    queryKey: ["config-schemas"],
    queryFn: () => getConfigSchemas() as Promise<Record<string, JsonSchema>>,
    staleTime: Infinity,
  })
}

export interface StarterTemplate {
  name: string
  type: string
  use_case: string
  description: string
  config: Record<string, unknown>
}

export function useStarterTemplates() {
  return useQuery<StarterTemplate[]>({
    queryKey: ["starter-templates"],
    queryFn: () => getStarterTemplates() as unknown as Promise<StarterTemplate[]>,
    staleTime: Infinity,
  })
}

export function useJobRecipes() {
  return useQuery<Job[]>({
    queryKey: ["job-recipes"],
    queryFn: async () => {
      const [sdg, training] = await Promise.all([
        getJobs({ status: "succeeded", type: "sdg" }),
        getJobs({ status: "succeeded", type: "training" }),
      ])
      return [...sdg, ...training].filter(
        (j) =>
          Object.keys(j.config ?? {}).length > 0 &&
          (j.config as Record<string, unknown>)?.mode !== "preview",
      )
    },
    staleTime: 30_000,
  })
}

export function useCreateJobFromConfig() {
  const queryClient = useQueryClient()

  return useMutation<
    Job,
    Error,
    { type: JobType; config: Record<string, unknown> }
  >({
    mutationFn: ({ type, config }) => {
      if (type === "training") return createTrainingJob(config)
      return createSdgJob(config)
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
      void queryClient.invalidateQueries({ queryKey: ["job-recipes"] })
    },
  })
}
