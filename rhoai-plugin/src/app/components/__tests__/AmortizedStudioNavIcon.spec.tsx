import React from 'react';
import { render } from '@testing-library/react';
import AmortizedStudioNavIcon from '../AmortizedStudioNavIcon';

describe('AmortizedStudioNavIcon (AmortizedStudioNavIcon) Component', () => {
  it('should render the SVG icon', () => {
    const { container } = render(<AmortizedStudioNavIcon />);
    const svg = container.querySelector('svg');
    expect(svg).toBeInTheDocument();
  });

  it('should use PatternFly SVG conventions', () => {
    const { container } = render(<AmortizedStudioNavIcon />);
    const svg = container.querySelector('svg');
    expect(svg).toHaveAttribute('width', '1em');
    expect(svg).toHaveAttribute('height', '1em');
    expect(svg).toHaveAttribute('viewBox', '0 0 32 32');
    expect(svg).toHaveClass('pf-v6-svg');
    expect(svg).toHaveAttribute('role', 'img');
    expect(svg).toHaveAttribute('aria-hidden', 'true');
  });

  it('should render the HW text', () => {
    const { container } = render(<AmortizedStudioNavIcon />);
    const textElement = container.querySelector('text');
    expect(textElement).toBeInTheDocument();
    expect(textElement?.textContent).toBe('HW');
  });

  it('should render the rectangle background with purple fill', () => {
    const { container } = render(<AmortizedStudioNavIcon />);
    const rect = container.querySelector('rect');
    expect(rect).toBeInTheDocument();
    expect(rect).toHaveAttribute('fill', '#6b21a8');
  });

  it('should have white bold centered text', () => {
    const { container } = render(<AmortizedStudioNavIcon />);
    const text = container.querySelector('text');
    expect(text).toHaveAttribute('fill', 'white');
    expect(text).toHaveAttribute('font-weight', 'bold');
    expect(text).toHaveAttribute('text-anchor', 'middle');
  });
});
