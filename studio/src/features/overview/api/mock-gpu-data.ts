import type { GpuUtilizationResponse } from "@/types/api"

export const MOCK_GPU_DATA: GpuUtilizationResponse = {
  nodes: [
    {
      index: 0,
      name: "NVIDIA A100-SXM4-80GB",
      utilization_pct: 42,
      memory_used_mb: 25600,
      memory_total_mb: 81920,
      temperature_c: 38,
    },
    {
      index: 1,
      name: "NVIDIA A100-SXM4-80GB",
      utilization_pct: 94,
      memory_used_mb: 58982,
      memory_total_mb: 81920,
      temperature_c: 71,
    },
  ],
}
