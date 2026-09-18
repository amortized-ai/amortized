import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router } from 'react-router-dom';
import '@patternfly/react-core/dist/styles/base.css';
import App from './app/App';

const root = ReactDOM.createRoot(document.getElementById('root')!);

root.render(
  <React.StrictMode>
    {/* [PLUGIN-SPECIFIC] basename must match route prefix — standalone dev only */}
    <Router basename="/amortized-studio">
      <App />
    </Router>
  </React.StrictMode>,
);
