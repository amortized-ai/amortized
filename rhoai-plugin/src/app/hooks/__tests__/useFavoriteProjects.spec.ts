import { renderHook, act } from '@testing-library/react';
import { useFavoriteProjects } from '../useFavoriteProjects';

const STORAGE_KEY = 'rhoai.project-favorites';

describe('useFavoriteProjects', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('returns empty favorites when localStorage is empty', () => {
    const { result } = renderHook(() => useFavoriteProjects());
    expect(result.current.favorites).toEqual([]);
  });

  it('reads existing favorites from localStorage', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['alpha', 'beta']));
    const { result } = renderHook(() => useFavoriteProjects());
    expect(result.current.favorites).toEqual(['alpha', 'beta']);
  });

  it('adds a favorite and persists to localStorage', () => {
    const { result } = renderHook(() => useFavoriteProjects());

    act(() => result.current.toggleFavorite('my-project'));

    expect(result.current.favorites).toEqual(['my-project']);
    expect(result.current.isFavorite('my-project')).toBe(true);
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual(['my-project']);
  });

  it('removes a favorite on second toggle', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['my-project']));
    const { result } = renderHook(() => useFavoriteProjects());

    act(() => result.current.toggleFavorite('my-project'));

    expect(result.current.favorites).toEqual([]);
    expect(result.current.isFavorite('my-project')).toBe(false);
    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toEqual([]);
  });

  it('handles corrupt localStorage gracefully', () => {
    localStorage.setItem(STORAGE_KEY, 'not-json');
    const { result } = renderHook(() => useFavoriteProjects());
    expect(result.current.favorites).toEqual([]);
  });

  it('handles non-array JSON gracefully', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ foo: 'bar' }));
    const { result } = renderHook(() => useFavoriteProjects());
    expect(result.current.favorites).toEqual([]);
  });

  it('filters out non-string values from localStorage', () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['valid', 42, null, 'also-valid']));
    const { result } = renderHook(() => useFavoriteProjects());
    expect(result.current.favorites).toEqual(['valid', 'also-valid']);
  });

  it('syncs when a storage event fires for the favorites key', () => {
    const { result } = renderHook(() => useFavoriteProjects());
    expect(result.current.favorites).toEqual([]);

    act(() => {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(['from-other-tab']));
      window.dispatchEvent(
        new StorageEvent('storage', {
          key: STORAGE_KEY,
          newValue: JSON.stringify(['from-other-tab']),
        }),
      );
    });

    expect(result.current.favorites).toEqual(['from-other-tab']);
  });

  it('ignores storage events for other keys', () => {
    const { result } = renderHook(() => useFavoriteProjects());

    act(() => {
      window.dispatchEvent(
        new StorageEvent('storage', {
          key: 'some-other-key',
          newValue: JSON.stringify(['should-not-appear']),
        }),
      );
    });

    expect(result.current.favorites).toEqual([]);
  });
});
