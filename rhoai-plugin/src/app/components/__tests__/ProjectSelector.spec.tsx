import { act, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useProjects } from '~/app/hooks/useProjects';
import { ProjectSelector } from '../ProjectSelector';

jest.mock('~/app/hooks/useProjects');
jest.mock('../CreateProjectModal', () => ({
  CreateProjectModal: ({ isOpen, onClose, onCreated }: { isOpen: boolean; onClose: () => void; onCreated: (name: string) => void }) =>
    isOpen ? (
      <div data-testid="create-project-modal">
        <button onClick={onClose}>Cancel</button>
        <button onClick={() => onCreated('new-project')}>Create</button>
      </div>
    ) : null,
}));

const userProjects = [
  { metadata: { name: 'my-app', uid: 'uid-1' } },
  { metadata: { name: 'staging', uid: 'uid-2' } },
  { metadata: { name: 'production', uid: 'uid-3' } },
];

const systemProjects = [
  { metadata: { name: 'openshift-monitoring', uid: 'uid-s1' } },
  { metadata: { name: 'kube-system', uid: 'uid-s2' } },
  { metadata: { name: 'default', uid: 'uid-s3' } },
];

const allProjects = [...userProjects, ...systemProjects];

const mockAddProject = jest.fn();

function mockProjects(projects = allProjects, loading = false, error: string | null = null) {
  (useProjects as jest.Mock).mockReturnValue({ projects, loading, error, refresh: jest.fn(), addProject: mockAddProject });
}

const STORAGE_KEY = 'rhoai.project-favorites';

describe('ProjectSelector', () => {
  const onSelect = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
    localStorage.clear();
  });

  it('shows placeholder when no project is selected', () => {
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);
    expect(screen.getByText('Select a project')).toBeInTheDocument();
  });

  it('shows selected project name in toggle', () => {
    mockProjects();
    render(<ProjectSelector selectedProject="my-app" onSelect={onSelect} />);
    expect(screen.getByText('Project: my-app')).toBeInTheDocument();
  });

  it('shows spinner when loading', () => {
    mockProjects([], true);
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);
    expect(screen.getByLabelText('Loading projects')).toBeInTheDocument();
  });

  it('shows error alert on failure', () => {
    mockProjects([], false, 'Network error');
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);
    expect(screen.getByText('Network error')).toBeInTheDocument();
  });

  it('opens dropdown and shows projects sorted alphabetically', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));

    const items = await screen.findAllByRole('menuitem');
    const names = items.map((item) => item.textContent).filter(Boolean);
    expect(names).toEqual(['my-app', 'production', 'staging']);
  });

  it('hides system namespaces by default', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));

    await waitFor(() => {
      expect(screen.queryByText('openshift-monitoring')).not.toBeInTheDocument();
      expect(screen.queryByText('kube-system')).not.toBeInTheDocument();
      expect(screen.queryByText('default')).not.toBeInTheDocument();
    });
  });

  it('shows system namespaces when toggle is turned on', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.click(screen.getByLabelText('Show default projects'));

    await waitFor(() => {
      expect(screen.getByText('openshift-monitoring')).toBeInTheDocument();
      expect(screen.getByText('kube-system')).toBeInTheDocument();
    });
  });

  it('does not show system namespace toggle when no system namespaces exist', async () => {
    const user = userEvent.setup();
    mockProjects(userProjects);
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));

    expect(screen.queryByText('Show default projects')).not.toBeInTheDocument();
  });

  it('filters projects by search text', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.type(screen.getByLabelText('Filter projects'), 'prod');

    await waitFor(() => {
      expect(screen.getByText('production')).toBeInTheDocument();
      expect(screen.queryByText('my-app')).not.toBeInTheDocument();
      expect(screen.queryByText('staging')).not.toBeInTheDocument();
    });
  });

  it('uses fuzzy matching for filter', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.type(screen.getByLabelText('Filter projects'), 'myap');

    await waitFor(() => {
      expect(screen.getByText('my-app')).toBeInTheDocument();
    });
  });

  it('shows empty state when filter matches nothing', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.type(screen.getByLabelText('Filter projects'), 'zzzzz');

    await waitFor(() => {
      expect(screen.getByText('No projects found')).toBeInTheDocument();
      expect(screen.getByText('Clear filters')).toBeInTheDocument();
    });
  });

  it('closes dropdown on window blur', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    expect(screen.getByText('Projects')).toBeInTheDocument();

    act(() => {
      window.dispatchEvent(new Event('blur'));
    });

    await waitFor(() => {
      expect(screen.queryByText('Projects')).not.toBeInTheDocument();
    });
  });

  it('calls onSelect when a project is clicked', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.click(await screen.findByText('staging'));

    expect(onSelect).toHaveBeenCalledWith('staging');
  });

  it('can be disabled', () => {
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} isDisabled />);
    expect(screen.getByLabelText('Select a project')).toBeDisabled();
  });

  it('shows favorited projects in the Favorites group', async () => {
    const user = userEvent.setup();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['staging']));
    mockProjects(userProjects);
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));

    expect(screen.getByText('Favorites')).toBeInTheDocument();
    expect(screen.getByText('Projects')).toBeInTheDocument();
  });

  it('toggles favorite when star icon is clicked', async () => {
    const user = userEvent.setup();
    mockProjects(userProjects);
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));

    const starButtons = screen.getAllByLabelText('not starred');
    await user.click(starButtons[0]);

    expect(JSON.parse(localStorage.getItem(STORAGE_KEY)!)).toContain('my-app');
  });

  it('filters favorites by search text', async () => {
    const user = userEvent.setup();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['my-app', 'staging']));
    mockProjects(userProjects);
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.type(screen.getByLabelText('Filter projects'), 'stag');

    await waitFor(() => {
      const items = screen.getAllByRole('menuitem');
      const names = items.map((item) => item.textContent);
      expect(names).toContain('staging');
      expect(names).not.toContain('my-app');
    });
  });

  it('hides Favorites group when no favorites match the filter', async () => {
    const user = userEvent.setup();
    localStorage.setItem(STORAGE_KEY, JSON.stringify(['my-app']));
    mockProjects(userProjects);
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.type(screen.getByLabelText('Filter projects'), 'prod');

    await waitFor(() => {
      expect(screen.queryByText('Favorites')).not.toBeInTheDocument();
      expect(screen.getByText('Projects')).toBeInTheDocument();
    });
  });

  it('shows Create project button in dropdown footer', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));

    expect(screen.getByText('Create project')).toBeInTheDocument();
  });

  it('opens CreateProjectModal when Create project is clicked', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.click(screen.getByText('Create project'));

    expect(screen.getByTestId('create-project-modal')).toBeInTheDocument();
  });

  it('auto-selects project after creation and adds it optimistically', async () => {
    const user = userEvent.setup();
    mockProjects();
    render(<ProjectSelector selectedProject={null} onSelect={onSelect} />);

    await user.click(screen.getByLabelText('Select a project'));
    await user.click(screen.getByText('Create project'));
    await user.click(screen.getByRole('button', { name: 'Create' }));

    expect(mockAddProject).toHaveBeenCalledWith({ metadata: { name: 'new-project', uid: '' } });
    expect(onSelect).toHaveBeenCalledWith('new-project');
  });
});
