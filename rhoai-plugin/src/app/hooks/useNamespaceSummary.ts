import { useState, useEffect, useCallback, useRef } from 'react';

export type PodCounts = {
  total: number;
  running: number;
  pending: number;
  succeeded: number;
  failed: number;
  unknown: number;
};

export type NamespaceInfo = {
  name: string;
  phase: string;
  pods: PodCounts;
};

export type NamespaceError = {
  name: string;
  error: string;
};

export type NamespaceSummaryData = {
  namespaces: NamespaceInfo[];
  errors: NamespaceError[];
};

export function useNamespaceSummary() {
  const [data, setData] = useState<NamespaceSummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(() => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);
    fetch('/amortized-studio/api/namespace-summary', { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch namespace summary: ${res.status}`);
        return res.json();
      })
      .then((json) => {
        setData(json);
        setLoading(false);
      })
      .catch((e) => {
        if (e.name === 'AbortError') return;
        setError(e.message);
        setLoading(false);
      });
  }, []);

  useEffect(() => {
    refresh();
    return () => controllerRef.current?.abort();
  }, [refresh]);

  return { data, loading, error, refresh };
}
