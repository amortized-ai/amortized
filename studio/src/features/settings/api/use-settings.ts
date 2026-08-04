import { useQuery } from "@tanstack/react-query"
import {
  getHealth,
  getConfig,
  getMlflowGatewayRoutes,
} from "@/lib/api-client"
import type {
  HealthResponse,
  ConfigResponse,
  MlflowGatewayRoute,
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
