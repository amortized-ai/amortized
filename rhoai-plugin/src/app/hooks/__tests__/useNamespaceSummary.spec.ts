import { renderHook, waitFor, act } from '@testing-library/react';
import { useNamespaceSummary } from '../useNamespaceSummary';

const mockSummary = {
  namespaces: [
    {
      name: 'project-a',
      phase: 'Active',
      pods: { total: 3, running: 2, pending: 1, succeeded: 0, failed: 0, unknown: 0 },
    },
    {
      name: 'project-b',
      phase: 'Active',
      pods: { total: 1, running: 1, pending: 0, succeeded: 0, failed: 0, unknown: 0 },
    },
  ],
};

describe('useNamespaceSummary', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should return namespace summary on success', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSummary),
    });

    const { result } = renderHook(() => useNamespaceSummary());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toEqual(mockSummary);
    expect(result.current.error).toBeNull();
    expect(global.fetch).toHaveBeenCalledWith(
      '/amortized-studio/api/namespace-summary',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('should return error on failure', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => useNamespaceSummary());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toBeNull();
    expect(result.current.error).toBe('Failed to fetch namespace summary: 500');
  });

  it('should support refresh', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockSummary),
    });

    const { result } = renderHook(() => useNamespaceSummary());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(global.fetch).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.refresh();
    });

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
});
