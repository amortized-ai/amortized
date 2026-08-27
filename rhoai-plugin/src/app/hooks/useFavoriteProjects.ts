import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'rhoai.project-favorites';

function readFavorites(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((v): v is string => typeof v === 'string') : [];
  } catch {
    return [];
  }
}

function writeFavorites(favorites: string[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(favorites));
}

export function useFavoriteProjects() {
  const [favorites, setFavorites] = useState<string[]>(readFavorites);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        setFavorites(readFavorites());
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const toggleFavorite = useCallback((name: string) => {
    setFavorites((prev) => {
      const next = prev.includes(name)
        ? prev.filter((f) => f !== name)
        : [...prev, name];
      writeFavorites(next);
      return next;
    });
  }, []);

  const isFavorite = useCallback(
    (name: string) => favorites.includes(name),
    [favorites],
  );

  return { favorites, toggleFavorite, isFavorite };
}
