import React from 'react';
import { act } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TextEncoder, TextDecoder } from 'util';

const originalAdvanceTimersByTime = jest.advanceTimersByTime.bind(jest);
Object.defineProperty(jest, 'advanceTimersByTime', {
  configurable: true,
  value: (ms: number) => {
    act(() => {
      originalAdvanceTimersByTime(ms);
    });
  },
});

global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder as unknown as typeof global.TextDecoder;

jest.mock('react-router-dom', () => {
  const actual = jest.requireActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => jest.fn(),
    useParams: () => ({}),
    useLocation: () => ({
      pathname: '/',
      search: '',
      state: undefined,
    }),
    Outlet: () => null,
    Navigate: ({ to }: { to: string }) => <div data-testid="navigate" data-to={to} />,
    Routes: ({ children }: { children: React.ReactNode }) => {
      const childrenArray = Array.isArray(children) ? children : [children];
      const firstChild = childrenArray[0] as React.ReactElement;
      return <div data-testid="routes">{firstChild}</div>;
    },
    Route: ({ element }: { element: React.ReactNode }) => element,
  };
});
