import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { toast } from "sonner"
import { getRecipes, getRecipe, submitRecipe, saveRecipe, deleteRecipe } from "@/lib/api-client"
import type { Recipe, Job } from "@/types/api"

export function useRecipes() {
  return useQuery<Recipe[]>({
    queryKey: ["recipes"],
    queryFn: getRecipes,
  })
}

export function useRecipe(name: string | null) {
  return useQuery<Recipe>({
    queryKey: ["recipes", name],
    queryFn: () => getRecipe(name!),
    enabled: !!name,
  })
}

export function useSaveRecipe() {
  const queryClient = useQueryClient()

  return useMutation<Recipe, Error, { name: string; type: string; description: string; config: Record<string, unknown> }>({
    mutationFn: ({ name, ...body }) => saveRecipe(name, body),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["recipes"] })
    },
  })
}

export function useExecuteRecipe() {
  const queryClient = useQueryClient()

  return useMutation<Job, Error, { recipe: string; overrides?: Record<string, unknown> }>({
    mutationFn: (data) => submitRecipe(data),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
  })
}


export function useDeleteRecipe() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (name: string) => deleteRecipe(name),
    onSuccess: () => {
      toast.success("Recipe deleted successfully")
    },
    onError: (err) => {
      toast.error(`Failed to delete recipe: ${err instanceof Error ? err.message : "Unknown error"}`)
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["recipes"] })
    },
  })
}
