import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import {
  Alert,
  Button,
  Divider,
  EmptyState,
  EmptyStateBody,
  EmptyStateFooter,
  EmptyStateActions,
  Menu,
  MenuContainer,
  MenuContent,
  MenuFooter,
  MenuGroup,
  MenuItem,
  MenuItemAction,
  MenuList,
  MenuSearch,
  MenuSearchInput,
  MenuToggle,
  Spinner,
  Switch,
  TextInput,
} from '@patternfly/react-core';
import { useProjects } from '~/app/hooks/useProjects';
import { useFavoriteProjects } from '~/app/hooks/useFavoriteProjects';
import { CreateProjectModal } from './CreateProjectModal';
import './ProjectSelector.css';

const SYSTEM_NAMESPACE_PREFIXES = ['openshift-', 'kube-'];
const SYSTEM_NAMESPACES = ['default', 'openshift'];

function isSystemNamespace(name: string): boolean {
  return (
    SYSTEM_NAMESPACES.includes(name) ||
    SYSTEM_NAMESPACE_PREFIXES.some((prefix) => name.startsWith(prefix))
  );
}

function fuzzysearch(needle: string, haystack: string): boolean {
  const nlen = needle.length;
  const hlen = haystack.length;
  if (nlen > hlen) return false;
  if (nlen === hlen) return needle === haystack;
  let j = 0;
  for (let i = 0; i < hlen && j < nlen; i++) {
    if (haystack[i] === needle[j]) j++;
  }
  return j === nlen;
}

export interface ProjectSelectorProps {
  selectedProject: string | null;
  onSelect: (project: string | null) => void;
  isDisabled?: boolean;
}

export const ProjectSelector: React.FC<ProjectSelectorProps> = ({
  selectedProject,
  onSelect,
  isDisabled = false,
}) => {
  const { projects, loading, error, addProject } = useProjects();
  const { isFavorite, toggleFavorite } = useFavoriteProjects();
  const menuRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const filterRef = useRef<HTMLInputElement>(null);

  const [isOpen, setIsOpen] = useState(false);
  const [filterText, setFilterText] = useState('');
  const [showSystemNamespaces, setShowSystemNamespaces] = useState(false);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    const close = () => {
      setIsOpen(false);
      setFilterText('');
    };
    window.addEventListener('blur', close);
    return () => window.removeEventListener('blur', close);
  }, [isOpen]);

  const sortedProjectNames = useMemo(
    () =>
      projects.map((p) => p.metadata.name).sort((a, b) => a.localeCompare(b)),
    [projects],
  );

  const hasSystemNamespaces = useMemo(
    () => sortedProjectNames.some(isSystemNamespace),
    [sortedProjectNames],
  );

  const { filteredFavorites, filteredProjects } = useMemo(() => {
    const needle = filterText.toLowerCase();
    const isVisible = (name: string) => {
      if (!showSystemNamespaces && isSystemNamespace(name)) return false;
      if (needle && !fuzzysearch(needle, name.toLowerCase())) return false;
      return true;
    };
    return sortedProjectNames.reduce(
      (acc, name) => {
        if (!isVisible(name)) return acc;
        if (isFavorite(name)) {
          acc.filteredFavorites.push(name);
        }
        acc.filteredProjects.push(name);
        return acc;
      },
      { filteredFavorites: [] as string[], filteredProjects: [] as string[] },
    );
  }, [sortedProjectNames, filterText, showSystemNamespaces, isFavorite]);

  const handleSelect = useCallback(
    (_event: React.MouseEvent | undefined, itemId: string | number | undefined) => {
      if (itemId != null) {
        onSelect(String(itemId));
      }
      setIsOpen(false);
      setFilterText('');
    },
    [onSelect],
  );

  const handleOpenChange = useCallback((open: boolean) => {
    setIsOpen(open);
    if (!open) {
      setFilterText('');
    }
  }, []);

  const handleClearFilter = useCallback(
    (event: React.MouseEvent) => {
      event.preventDefault();
      event.stopPropagation();
      setFilterText('');
      filterRef.current?.focus();
    },
    [],
  );

  const handleProjectCreated = useCallback(
    (projectName: string) => {
      setIsCreateModalOpen(false);
      addProject({ metadata: { name: projectName, uid: '' } });
      onSelect(projectName);
    },
    [addProject, onSelect],
  );

  if (loading) {
    return <Spinner size="md" aria-label="Loading projects" />;
  }

  if (error) {
    return (
      <Alert variant="danger" title="Failed to load projects" isInline>
        {error}
      </Alert>
    );
  }

  const title = selectedProject
    ? `Project: ${selectedProject}`
    : 'Select a project';

  const toggle = (
    <MenuToggle
      ref={toggleRef}
      onClick={() => setIsOpen(!isOpen)}
      isExpanded={isOpen}
      isDisabled={isDisabled}
      className="project-selector__toggle"
      aria-label="Select a project"
    >
      {title}
    </MenuToggle>
  );

  const renderMenuItem = (name: string) => (
    <MenuItem
      key={name}
      itemId={name}
      isSelected={selectedProject === name}
      actions={
        <MenuItemAction
          icon="favorites"
          isFavorited={isFavorite(name)}
          actionId="fav"
          aria-label={isFavorite(name) ? 'starred' : 'not starred'}
          onClick={() => toggleFavorite(name)}
        />
      }
    >
      {name}
    </MenuItem>
  );

  const menu = (
    <Menu
      ref={menuRef}
      onSelect={handleSelect}
      activeItemId={selectedProject ?? undefined}
      isScrollable
    >
      <MenuContent maxMenuHeight="60vh">
        <MenuSearch>
          <MenuSearchInput>
            <TextInput
              ref={filterRef}
              value={filterText}
              aria-label="Filter projects"
              type="search"
              placeholder="Filter by name..."
              onChange={(_, value) => setFilterText(value)}
            />
          </MenuSearchInput>
        </MenuSearch>
        {filteredProjects.length === 0 ? (
          <>
            <Divider />
            <EmptyState headingLevel="h4" titleText="No projects found">
              <EmptyStateBody>
                No results match the filter criteria.
              </EmptyStateBody>
              <EmptyStateFooter>
                <EmptyStateActions>
                  <Button variant="link" onClick={handleClearFilter}>
                    Clear filters
                  </Button>
                </EmptyStateActions>
              </EmptyStateFooter>
            </EmptyState>
          </>
        ) : (
          <>
            {filteredFavorites.length > 0 && (
              <>
                <Divider />
                <MenuGroup label="Favorites">
                  <MenuList>
                    {filteredFavorites.map(renderMenuItem)}
                  </MenuList>
                </MenuGroup>
              </>
            )}
            {hasSystemNamespaces && (
              <>
                <Divider />
                <MenuSearch>
                  <MenuSearchInput>
                    <Switch
                      label="Show default projects"
                      isChecked={showSystemNamespaces}
                      onChange={(_, checked) => setShowSystemNamespaces(checked)}
                      isReversed
                    />
                  </MenuSearchInput>
                </MenuSearch>
                <Divider />
              </>
            )}
            {!hasSystemNamespaces && <Divider />}
            <MenuGroup label="Projects">
              <MenuList>
                {filteredProjects.map(renderMenuItem)}
              </MenuList>
            </MenuGroup>
          </>
        )}
      </MenuContent>
      <MenuFooter>
        <Button
          variant="link"
          isInline
          onClick={() => {
            setIsOpen(false);
            setFilterText('');
            setIsCreateModalOpen(true);
          }}
        >
          Create project
        </Button>
      </MenuFooter>
    </Menu>
  );

  return (
    <>
      <MenuContainer
        isOpen={isOpen}
        onOpenChange={handleOpenChange}
        menu={menu}
        menuRef={menuRef}
        toggle={toggle}
        toggleRef={toggleRef}
      />
      <CreateProjectModal
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onCreated={handleProjectCreated}
      />
    </>
  );
};
