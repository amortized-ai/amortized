import { useState, useEffect, useCallback, useRef } from 'react';

export type K8sResource = {
  apiVersion: string;
  kind: string;
  metadata: {
    name: string;
    namespace: string;
    uid: string;
    creationTimestamp: string;
    labels?: Record<string, string>;
  };
  spec?: Record<string, unknown>;
  status?: Record<string, unknown>;
};

export function useK8sResources<T extends K8sResource = K8sResource>(
  apiPath: string | null,
) {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    if (!apiPath) {
      setItems([]);
      setLoading(false);
      return;
    }

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);
    fetch(`/api/k8s${apiPath}`, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        setItems(data.items ?? []);
        setLoading(false);
      })
      .catch((e) => {
        if (e.name === 'AbortError') return;
        setError(e.message);
        setLoading(false);
      });
  }, [apiPath]);

  useEffect(() => {
    refresh();
    return () => controllerRef.current?.abort();
  }, [refresh]);

  return { items, loading, error, refresh };
}

export async function createK8sResource(
  apiPath: string,
  resource: Record<string, unknown>,
): Promise<K8sResource> {
  const response = await fetch(`/api/k8s${apiPath}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(resource),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(
      err.message || `Failed to create resource: ${response.status}`,
    );
  }
  return response.json();
}

export async function deleteK8sResource(apiPath: string): Promise<void> {
  const response = await fetch(`/api/k8s${apiPath}`, { method: 'DELETE' });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(
      err.message || `Failed to delete resource: ${response.status}`,
    );
  }
}
