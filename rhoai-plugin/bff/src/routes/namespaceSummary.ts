import { Request, Response } from 'express';
import { k8sRequest } from '../utils/k8sClient';
import { K8sResource, K8sList, PodCounts, NamespaceInfo, NamespaceError, NamespaceSummaryResponse } from '../types';

function countPods(items: K8sResource[]): PodCounts {
  const counts: PodCounts = {
    total: items.length,
    running: 0,
    pending: 0,
    succeeded: 0,
    failed: 0,
    unknown: 0,
  };

  for (const item of items) {
    const phase = item.status?.phase;
    switch (phase) {
      case 'Running':
        counts.running++;
        break;
      case 'Pending':
        counts.pending++;
        break;
      case 'Succeeded':
        counts.succeeded++;
        break;
      case 'Failed':
        counts.failed++;
        break;
      default:
        counts.unknown++;
        break;
    }
  }

  return counts;
}

export async function namespaceSummaryHandler(
  req: Request,
  res: Response,
): Promise<void> {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    res.status(401).json({ error: 'Missing or invalid Authorization header' });
    return;
  }

  const token = authHeader.slice(7);

  try {
    const projectsData = await k8sRequest<K8sList>(
      token,
      '/apis/project.openshift.io/v1/projects',
    );

    const results = await Promise.allSettled(
      projectsData.items.map(async (project) => {
        const name = project.metadata.name;
        const phase = project.status?.phase || 'Active';
        const podsData = await k8sRequest<K8sList>(
          token,
          `/api/v1/namespaces/${name}/pods`,
        );
        return {
          name,
          phase,
          pods: countPods(podsData.items || []),
        } as NamespaceInfo;
      }),
    );

    const namespaces: NamespaceInfo[] = [];
    const errors: NamespaceError[] = [];

    results.forEach((r, i) => {
      if (r.status === 'fulfilled') {
        namespaces.push(r.value);
      } else {
        const name = projectsData.items[i].metadata.name;
        errors.push({ name, error: r.reason?.message ?? 'Unknown error' });
      }
    });

    const response: NamespaceSummaryResponse = { namespaces, errors };
    res.json(response);
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    console.error('Namespace summary error:', message);
    res.status(502).json({ error: message });
  }
}
