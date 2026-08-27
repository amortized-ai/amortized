import { useState, useCallback, useEffect } from 'react';

const STORAGE_KEY = 'rhoai.last-selected-project';

function readLastProject(): string | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return typeof raw === 'string' && raw.length > 0 ? raw : null;
  } catch {
    return null;
  }
}

function writeLastProject(project: string | null): void {
  if (project === null) {
    localStorage.removeItem(STORAGE_KEY);
  } else {
    localStorage.setItem(STORAGE_KEY, project);
  }
}

export function useLastSelectedProject(): [string | null, (project: string | null) => void] {
  const [selected, setSelected] = useState<string | null>(readLastProject);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        setSelected(readLastProject());
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, []);

  const select = useCallback((project: string | null) => {
    writeLastProject(project);
    setSelected(project);
  }, []);

  return [selected, select];
}
