import { useState, useEffect, useCallback, useRef } from 'react';

export type Project = {
  metadata: {
    name: string;
    uid: string;
    labels?: Record<string, string>;
    annotations?: Record<string, string>;
  };
  status?: {
    phase: string;
  };
};

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback((): Promise<Project[]> => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    setLoading(true);
    setError(null);
    return fetch('/api/k8s/apis/project.openshift.io/v1/projects', { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch projects: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        const items: Project[] = data.items ?? [];
        setProjects(items);
        setLoading(false);
        return items;
      })
      .catch((e) => {
        if (e.name === 'AbortError') return [];
        setError(e.message);
        setLoading(false);
        return [];
      });
  }, []);

  const addProject = useCallback((project: Project) => {
    setProjects((prev) => {
      if (prev.some((p) => p.metadata.name === project.metadata.name)) return prev;
      return [...prev, project];
    });
  }, []);

  useEffect(() => {
    refresh();
    return () => controllerRef.current?.abort();
  }, [refresh]);

  return { projects, loading, error, refresh, addProject };
}
