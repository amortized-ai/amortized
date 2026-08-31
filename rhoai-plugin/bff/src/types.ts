export interface K8sMetadata {
  name: string;
  namespace?: string;
  [key: string]: unknown;
}

export interface K8sResource {
  metadata: K8sMetadata;
  status?: { phase?: string; [key: string]: unknown };
  [key: string]: unknown;
}

export interface K8sList {
  items: K8sResource[];
}

export interface PodCounts {
  total: number;
  running: number;
  pending: number;
  succeeded: number;
  failed: number;
  unknown: number;
}

export interface NamespaceInfo {
  name: string;
  phase: string;
  pods: PodCounts;
}

export interface NamespaceError {
  name: string;
  error: string;
}

export interface NamespaceSummaryResponse {
  namespaces: NamespaceInfo[];
  errors: NamespaceError[];
}
