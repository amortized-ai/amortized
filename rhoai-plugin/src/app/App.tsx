import React from 'react';
import { Routes, Route } from 'react-router-dom';
import CommunityBanner from './components/CommunityBanner';
import StudioEmbed from './components/StudioEmbed';

// The studio is a full application with its own internal routing, so the plugin
// mounts it once at the section root and lets the embedded SPA own navigation
// below that. The dashboard chrome (nav sidebar + header) stays native; the
// content area is the studio.
const App: React.FC = () => (
  <div
    className="amortized-studio-layout"
    style={{ display: 'flex', flexDirection: 'column', height: '100%' }}
  >
    {/* [SHARED] Do not remove — all community plugins must display the CommunityBanner */}
    <CommunityBanner />
    <div
      className="amortized-studio-content"
      style={{ flex: 1, minHeight: 0, display: 'flex' }}
    >
      <Routes>
        <Route path="*" element={<StudioEmbed />} />
      </Routes>
    </div>
  </div>
);

export default App;
