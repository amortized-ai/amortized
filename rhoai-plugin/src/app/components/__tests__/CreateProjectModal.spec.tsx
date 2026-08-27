import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CreateProjectModal, toResourceName, validateResourceName } from '../CreateProjectModal';
import { createK8sResource } from '~/app/hooks/useK8sResources';

jest.mock('~/app/hooks/useK8sResources', () => ({
  createK8sResource: jest.fn(),
}));

describe('toResourceName', () => {
  it('converts display name to resource name', () => {
    expect(toResourceName('My Cool Project')).toBe('my-cool-project');
  });

  it('collapses multiple hyphens', () => {
    expect(toResourceName('my---project')).toBe('my-project');
  });

  it('trims leading and trailing hyphens', () => {
    expect(toResourceName('--my-project--')).toBe('my-project');
  });

  it('truncates to 30 characters', () => {
    const long = 'a'.repeat(50);
    expect(toResourceName(long)).toHaveLength(30);
  });

  it('replaces non-alphanumeric characters with hyphens', () => {
    expect(toResourceName('Amortized Studio! @#$%')).toBe('amortized-studio');
  });
});

describe('validateResourceName', () => {
  it('accepts valid names', () => {
    const result = validateResourceName('my-project');
    expect(result.lengthStatus).toBe('success');
    expect(result.formatStatus).toBe('success');
  });

  it('rejects names exceeding 30 characters', () => {
    const result = validateResourceName('a'.repeat(31));
    expect(result.lengthStatus).toBe('error');
  });

  it('rejects names starting with a hyphen', () => {
    const result = validateResourceName('-my-project');
    expect(result.formatStatus).toBe('error');
  });

  it('rejects names ending with a hyphen', () => {
    const result = validateResourceName('my-project-');
    expect(result.formatStatus).toBe('error');
  });

  it('rejects names with uppercase letters', () => {
    const result = validateResourceName('MyProject');
    expect(result.formatStatus).toBe('error');
  });

  it('returns default for empty names', () => {
    const result = validateResourceName('');
    expect(result.lengthStatus).toBe('default');
    expect(result.formatStatus).toBe('default');
  });
});

describe('CreateProjectModal', () => {
  const onClose = jest.fn();
  const onCreated = jest.fn();

  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('renders form fields when open', () => {
    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);
    expect(screen.getByText('Create project')).toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Name' })).toBeInTheDocument();
    expect(screen.getByText('Edit resource name')).toBeInTheDocument();
  });

  it('shows resource name field when Edit resource name is clicked', async () => {
    const user = userEvent.setup();
    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.click(screen.getByText('Edit resource name'));

    expect(screen.getByRole('textbox', { name: 'Resource name' })).toBeInTheDocument();
  });

  it('allows manual resource name editing', async () => {
    const user = userEvent.setup();
    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'My App');
    await user.click(screen.getByText('Edit resource name'));

    const resourceInput = screen.getByRole('textbox', { name: 'Resource name' });
    expect(resourceInput).toHaveValue('my-app');

    await user.clear(resourceInput);
    await user.type(resourceInput, 'custom-name');

    expect(resourceInput).toHaveValue('custom-name');
  });

  it('creates project and calls onCreated on success', async () => {
    const user = userEvent.setup();
    (createK8sResource as jest.Mock).mockResolvedValue({});
    global.fetch = jest.fn().mockResolvedValue({ ok: true });

    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Test Project');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(createK8sResource).toHaveBeenCalledWith(
        '/apis/project.openshift.io/v1/projectrequests',
        expect.objectContaining({
          kind: 'ProjectRequest',
          metadata: { name: 'test-project' },
          displayName: 'Test Project',
        }),
      );
    });

    expect(onCreated).toHaveBeenCalledWith('test-project');
  });

  it('sends PATCH to label the namespace', async () => {
    const user = userEvent.setup();
    (createK8sResource as jest.Mock).mockResolvedValue({});
    global.fetch = jest.fn().mockResolvedValue({ ok: true });

    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Labeled Project');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/k8s/api/v1/namespaces/labeled-project',
        expect.objectContaining({
          method: 'PATCH',
          headers: { 'Content-Type': 'application/merge-patch+json' },
        }),
      );
    });
  });

  it('shows error on API failure', async () => {
    const user = userEvent.setup();
    (createK8sResource as jest.Mock).mockRejectedValue(new Error('Conflict'));

    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Fail Project');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(screen.getByText('Conflict')).toBeInTheDocument();
    });

    expect(onCreated).not.toHaveBeenCalled();
  });

  it('calls onClose and resets form on cancel', async () => {
    const user = userEvent.setup();
    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Some Name');
    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(onClose).toHaveBeenCalled();
  });

  it('submits with description when provided', async () => {
    const user = userEvent.setup();
    (createK8sResource as jest.Mock).mockResolvedValue({});
    global.fetch = jest.fn().mockResolvedValue({ ok: true });

    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);

    await user.type(screen.getByRole('textbox', { name: 'Name' }), 'Described Project');
    await user.type(screen.getByRole('textbox', { name: 'Description' }), 'A test description');
    await user.click(screen.getByRole('button', { name: 'Create' }));

    await waitFor(() => {
      expect(createK8sResource).toHaveBeenCalledWith(
        '/apis/project.openshift.io/v1/projectrequests',
        expect.objectContaining({
          description: 'A test description',
        }),
      );
    });
  });

  it('disables Create button when name is empty', () => {
    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);
    expect(screen.getByRole('button', { name: 'Create' })).toBeDisabled();
  });

  it('shows help popover icon', () => {
    render(<CreateProjectModal isOpen onClose={onClose} onCreated={onCreated} />);
    expect(screen.getByLabelText('Resource name help')).toBeInTheDocument();
  });
});
