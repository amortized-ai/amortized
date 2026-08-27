import { renderHook, act } from '@testing-library/react';
import { useLastSelectedProject } from '../useLastSelectedProject';

const STORAGE_KEY = 'rhoai.last-selected-project';

describe('useLastSelectedProject', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('returns null when localStorage is empty', () => {
    const { result } = renderHook(() => useLastSelectedProject());
    expect(result.current[0]).toBeNull();
  });

  it('restores saved project on mount', () => {
    localStorage.setItem(STORAGE_KEY, 'my-project');
    const { result } = renderHook(() => useLastSelectedProject());
    expect(result.current[0]).toBe('my-project');
  });

  it('persists to localStorage on select', () => {
    const { result } = renderHook(() => useLastSelectedProject());

    act(() => result.current[1]('new-project'));

    expect(result.current[0]).toBe('new-project');
    expect(localStorage.getItem(STORAGE_KEY)).toBe('new-project');
  });

  it('removes from localStorage when set to null', () => {
    localStorage.setItem(STORAGE_KEY, 'old-project');
    const { result } = renderHook(() => useLastSelectedProject());

    act(() => result.current[1](null));

    expect(result.current[0]).toBeNull();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it('syncs when a storage event fires for the key', () => {
    const { result } = renderHook(() => useLastSelectedProject());
    expect(result.current[0]).toBeNull();

    act(() => {
      localStorage.setItem(STORAGE_KEY, 'from-other-tab');
      window.dispatchEvent(
        new StorageEvent('storage', {
          key: STORAGE_KEY,
          newValue: 'from-other-tab',
        }),
      );
    });

    expect(result.current[0]).toBe('from-other-tab');
  });

  it('ignores storage events for other keys', () => {
    const { result } = renderHook(() => useLastSelectedProject());

    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', {
          key: 'some-other-key',
          newValue: 'should-not-appear',
        }),
      );
    });

    expect(result.current[0]).toBeNull();
  });
});
