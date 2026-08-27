import { renderHook, waitFor } from '@testing-library/react';
import { useK8sResources, createK8sResource, deleteK8sResource } from '../useK8sResources';

const mockDeployments = [
  {
    apiVersion: 'apps/v1',
    kind: 'Deployment',
    metadata: {
      name: 'test-deploy',
      namespace: 'test-ns',
      uid: 'uid-1',
      creationTimestamp: '2026-01-01T00:00:00Z',
    },
    spec: { replicas: 1 },
    status: { readyReplicas: 1 },
  },
];

describe('useK8sResources', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should return empty items when apiPath is null', () => {
    const { result } = renderHook(() => useK8sResources(null));
    expect(result.current.items).toEqual([]);
    expect(result.current.loading).toBe(false);
  });

  it('should fetch resources on mount', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: mockDeployments }),
    });

    const { result } = renderHook(() =>
      useK8sResources('/apis/apps/v1/namespaces/test-ns/deployments'),
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.items).toEqual(mockDeployments);
    expect(result.current.error).toBeNull();
  });

  it('should handle fetch error', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
    });

    const { result } = renderHook(() =>
      useK8sResources('/apis/apps/v1/namespaces/test-ns/deployments'),
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.items).toEqual([]);
    expect(result.current.error).toBe('404 Not Found');
  });
});

describe('createK8sResource', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should POST the resource', async () => {
    const created = { ...mockDeployments[0] };
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(created),
    });

    const result = await createK8sResource(
      '/apis/apps/v1/namespaces/test-ns/deployments',
      { apiVersion: 'apps/v1', kind: 'Deployment', metadata: { name: 'test' } },
    );

    expect(result).toEqual(created);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/k8s/apis/apps/v1/namespaces/test-ns/deployments',
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('should throw on failure', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 409,
      json: () => Promise.resolve({ message: 'already exists' }),
    });

    await expect(
      createK8sResource('/apis/apps/v1/namespaces/test-ns/deployments', {}),
    ).rejects.toThrow('already exists');
  });
});

describe('deleteK8sResource', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should DELETE the resource', async () => {
    global.fetch = jest.fn().mockResolvedValue({ ok: true });

    await deleteK8sResource('/apis/apps/v1/namespaces/test-ns/deployments/test');

    expect(global.fetch).toHaveBeenCalledWith(
      '/api/k8s/apis/apps/v1/namespaces/test-ns/deployments/test',
      { method: 'DELETE' },
    );
  });

  it('should throw on failure', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 403,
      json: () => Promise.resolve({ message: 'forbidden' }),
    });

    await expect(
      deleteK8sResource('/apis/apps/v1/namespaces/test-ns/deployments/test'),
    ).rejects.toThrow('forbidden');
  });
});
