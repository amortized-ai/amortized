import { useState, useEffect } from 'react';

export type KubeStatus = {
  currentContext: string;
  currentUser: string;
  namespace: string;
  userName: string;
  userID: string;
  clusterID: string;
  clusterBranding: string;
  isAdmin: boolean;
  isAllowed: boolean;
  serverURL: string;
};

export function useCurrentUser() {
  const [user, setUser] = useState<KubeStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/status', { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to fetch status: ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setUser(data.kube);
        setLoading(false);
      })
      .catch((e) => {
        if (e.name === 'AbortError') return;
        setError(e.message);
        setLoading(false);
      });
    return () => controller.abort();
  }, []);

  return { user, loading, error };
}
