import {
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query"
import {
  getHealth,
  getConfig,
  getMlflowGatewayRoutes,
  createMlflowGatewayRoute,
  deleteMlflowGatewayRoute,
  getMlflowGatewayConnections,
  createMlflowGatewayConnection,
  deleteMlflowGatewayConnection,
} from "@/lib/api-client"
import type {
  HealthResponse,
  ConfigResponse,
  MlflowGatewayRoute,
  MlflowGatewayRouteCreate,
  MlflowGatewayConnection,
  MlflowGatewayConnectionCreate,
} from "@/types/api"

export function useHealth(options?: { refetchInterval?: number }) {
  return useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: options?.refetchInterval,
    refetchOnWindowFocus: true,
  })
}

export function useConfig() {
  return useQuery<ConfigResponse>({
    queryKey: ["config"],
    queryFn: getConfig,
  })
}

export function useGatewayRoutes() {
  return useQuery<MlflowGatewayRoute[]>({
    queryKey: ["mlflow", "gateway", "routes"],
    queryFn: async () => {
      const resp = await getMlflowGatewayRoutes()
      return resp.routes ?? []
    },
  })
}

export function useCreateGatewayRoute() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: MlflowGatewayRouteCreate) => createMlflowGatewayRoute(data),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["mlflow", "gateway", "routes"] })
    },
  })
}

export function useDeleteGatewayRoute() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (name: string) => deleteMlflowGatewayRoute(name),
    onMutate: async (name) => {
      await queryClient.cancelQueries({ queryKey: ["mlflow", "gateway", "routes"] })

      const previous = queryClient.getQueryData<MlflowGatewayRoute[]>(["mlflow", "gateway", "routes"])

      queryClient.setQueryData<MlflowGatewayRoute[]>(
        ["mlflow", "gateway", "routes"],
        (old) => old?.filter((r) => r.name !== name),
      )

      return { previous }
    },
    onError: (_err, _name, context) => {
      if (context?.previous) {
        queryClient.setQueryData<MlflowGatewayRoute[]>(
          ["mlflow", "gateway", "routes"],
          context.previous,
        )
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["mlflow", "gateway", "routes"] })
    },
  })
}

export function useGatewayConnections() {
  return useQuery<MlflowGatewayConnection[]>({
    queryKey: ["mlflow", "gateway", "connections"],
    queryFn: getMlflowGatewayConnections,
  })
}

export function useCreateGatewayConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (data: MlflowGatewayConnectionCreate) => createMlflowGatewayConnection(data),
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["mlflow", "gateway", "connections"] })
    },
  })
}

export function useDeleteGatewayConnection() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (secretId: string) => deleteMlflowGatewayConnection(secretId),
    onMutate: async (secretId) => {
      await queryClient.cancelQueries({ queryKey: ["mlflow", "gateway", "connections"] })

      const previous = queryClient.getQueryData<MlflowGatewayConnection[]>(["mlflow", "gateway", "connections"])

      queryClient.setQueryData<MlflowGatewayConnection[]>(
        ["mlflow", "gateway", "connections"],
        (old) => old?.filter((c) => c.secret_id !== secretId),
      )

      return { previous }
    },
    onError: (_err, _id, context) => {
      if (context?.previous) {
        queryClient.setQueryData<MlflowGatewayConnection[]>(
          ["mlflow", "gateway", "connections"],
          context.previous,
        )
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ["mlflow", "gateway", "connections"] })
    },
  })
}
