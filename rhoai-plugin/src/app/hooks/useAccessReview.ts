import { useState, useEffect } from 'react';

export type AccessReviewResult = {
  verb: string;
  resource: string;
  group: string;
  allowed: boolean;
};

async function checkAccess(
  namespace: string,
  verb: string,
  group: string,
  resource: string,
): Promise<boolean> {
  const review = {
    apiVersion: 'authorization.k8s.io/v1',
    kind: 'SelfSubjectAccessReview',
    spec: {
      resourceAttributes: { namespace, verb, group, resource },
    },
  };

  const response = await fetch(
    '/api/k8s/apis/authorization.k8s.io/v1/selfsubjectaccessreviews',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(review),
    },
  );

  if (!response.ok) {
    throw new Error(`Access review failed: ${response.status}`);
  }

  const result = await response.json();
  return result.status?.allowed === true;
}

type ResourceCheck = { group: string; resource: string };

const DEFAULT_CHECKS: ResourceCheck[] = [
  { group: 'apps', resource: 'deployments' },
  { group: '', resource: 'services' },
  { group: '', resource: 'configmaps' },
  { group: '', resource: 'secrets' },
];

const VERBS = ['get', 'list', 'create', 'delete'];

export function useAccessReview(namespace: string | null) {
  const [results, setResults] = useState<AccessReviewResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!namespace) {
      setResults([]);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    const checks = DEFAULT_CHECKS.flatMap((check) =>
      VERBS.map((verb) => ({ verb, ...check })),
    );

    Promise.allSettled(
      checks.map(({ verb, group, resource }) =>
        checkAccess(namespace, verb, group, resource).then((allowed) => ({
          verb,
          group,
          resource,
          allowed,
        })),
      ),
    )
      .then((settled) => {
        if (cancelled) return;
        const succeeded: AccessReviewResult[] = [];
        const errors: string[] = [];
        settled.forEach((r) => {
          if (r.status === 'fulfilled') {
            succeeded.push(r.value);
          } else {
            errors.push(r.reason?.message ?? 'Unknown error');
          }
        });
        setResults(succeeded);
        if (errors.length > 0 && succeeded.length === 0) {
          setError(errors[0]);
        }
        setLoading(false);
      });

    return () => { cancelled = true; };
  }, [namespace]);

  return { results, loading, error };
}
