import { renderHook, waitFor } from '@testing-library/react';
import { useAccessReview } from '../useAccessReview';

describe('useAccessReview', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should return empty results when namespace is null', () => {
    const { result } = renderHook(() => useAccessReview(null));
    expect(result.current.results).toEqual([]);
    expect(result.current.loading).toBe(false);
  });

  it('should check permissions for the given namespace', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ status: { allowed: true } }),
    });

    const { result } = renderHook(() => useAccessReview('test-ns'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.results.length).toBeGreaterThan(0);
    expect(result.current.results.every((r) => r.allowed === true)).toBe(true);
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/k8s/apis/authorization.k8s.io/v1/selfsubjectaccessreviews',
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('should set error when API returns non-ok response', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => useAccessReview('test-ns'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBe('Access review failed: 500');
    expect(result.current.results).toEqual([]);
  });

  it('should handle denied permissions', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ status: { allowed: false } }),
    });

    const { result } = renderHook(() => useAccessReview('test-ns'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.results.every((r) => r.allowed === false)).toBe(true);
  });
});
