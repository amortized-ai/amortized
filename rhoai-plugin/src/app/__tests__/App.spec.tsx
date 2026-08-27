import { render, screen } from '@testing-library/react';
import App from '../App';

jest.mock('../pages/UserInfoPage', () => {
  const MockPage = () => <div data-testid="user-info-page">User Info Page</div>;
  MockPage.displayName = 'MockUserInfoPage';
  return { __esModule: true, default: MockPage };
});

jest.mock('../pages/ClusterResourcesPage', () => {
  const MockPage = () => <div data-testid="cluster-resources-page">Cluster Resources Page</div>;
  MockPage.displayName = 'MockClusterResourcesPage';
  return { __esModule: true, default: MockPage };
});

jest.mock('../pages/NamespaceSummaryPage', () => {
  const MockPage = () => <div data-testid="namespace-summary-page">Namespace Summary Page</div>;
  MockPage.displayName = 'MockNamespaceSummaryPage';
  return { __esModule: true, default: MockPage };
});

describe('App Component', () => {
  it('should render the first route element', () => {
    render(<App />);
    expect(screen.getByTestId('routes')).toBeInTheDocument();
  });
});
