import { render } from '@testing-library/react';
import App from '../App';

// StudioEmbed renders an <iframe>; jsdom supports it but we only need to assert
// the App mounts and embeds the studio.
describe('App Component', () => {
  it('renders the studio embed iframe', () => {
    const { container } = render(<App />);
    const iframe = container.querySelector('iframe');
    expect(iframe).toBeInTheDocument();
    expect(iframe).toHaveAttribute('title', 'Amortized Studio');
  });
});
