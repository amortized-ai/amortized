import { renderHook, waitFor, act } from '@testing-library/react';
import { useProjects } from '../useProjects';

const mockProjects = [
  { metadata: { name: 'project-a', uid: 'uid-a' }, status: { phase: 'Active' } },
  { metadata: { name: 'project-b', uid: 'uid-b' }, status: { phase: 'Active' } },
];

describe('useProjects', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('should return projects on success', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: mockProjects }),
    });

    const { result } = renderHook(() => useProjects());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.projects).toEqual(mockProjects);
    expect(result.current.error).toBeNull();
    expect(global.fetch).toHaveBeenCalledWith(
      '/api/k8s/apis/project.openshift.io/v1/projects',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('should return error on failure', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => useProjects());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.projects).toEqual([]);
    expect(result.current.error).toBe('Failed to fetch projects: 500');
  });

  it('should optimistically add a project', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: mockProjects }),
    });

    const { result } = renderHook(() => useProjects());

    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.addProject({ metadata: { name: 'new-project', uid: '' } });
    });

    expect(result.current.projects).toHaveLength(3);
    expect(result.current.projects[2].metadata.name).toBe('new-project');
  });

  it('should not duplicate when adding an existing project', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: mockProjects }),
    });

    const { result } = renderHook(() => useProjects());

    await waitFor(() => expect(result.current.loading).toBe(false));

    act(() => {
      result.current.addProject({ metadata: { name: 'project-a', uid: 'uid-a' } });
    });

    expect(result.current.projects).toHaveLength(2);
  });

  it('should support refresh', async () => {
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ items: mockProjects }),
    });

    const { result } = renderHook(() => useProjects());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(global.fetch).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.refresh();
    });

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(global.fetch).toHaveBeenCalledTimes(2);
  });
});
