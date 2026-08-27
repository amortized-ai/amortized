import { renderHook, waitFor } from '@testing-library/react';
import { useCurrentUser } from '../useCurrentUser';

const mockUser = {
  userName: 'test-user',
  userID: 'uid-123',
  isAdmin: false,
  clusterID: 'cluster-abc',
  clusterBranding: 'ocp',
  namespace: 'default',
  currentContext: 'ctx',
  currentUser: 'test-user',
  isAllowed: true,
  serverURL: 'https://api.cluster.example.com:6443',
};

describe('useCurrentUser', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should return user data on success', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ kube: mockUser }),
    });

    const { result } = renderHook(() => useCurrentUser());

    expect(result.current.loading).toBe(true);

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.user).toEqual(mockUser);
    expect(result.current.error).toBeNull();
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/status',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('should return error on fetch failure', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 403,
    });

    const { result } = renderHook(() => useCurrentUser());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.user).toBeNull();
    expect(result.current.error).toBe('Failed to fetch status: 403');
  });

  it('should return error on network failure', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('Network error'));

    const { result } = renderHook(() => useCurrentUser());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.user).toBeNull();
    expect(result.current.error).toBe('Network error');
  });
});
