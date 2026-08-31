import React from 'react';

/**
 * Same-origin embed of the Amortized Studio SPA.
 *
 * The studio is served THROUGH the RHOAI dashboard proxy (a federation `proxy`
 * entry with authorize:true → the studio-gateway service). Because the iframe
 * src is a dashboard-origin path, it inherits the dashboard's OpenShift session
 * — no separate OAuth redirect (which the OAuth server blocks in an iframe via
 * X-Frame-Options), no third-party cookies.
 *
 * The studio SPA is built with base path = this same prefix so its assets and
 * API calls resolve under it; the gateway strips the prefix and routes per-user.
 *
 * Path override (build time): __STUDIO_EMBED_PATH__ (BuildConfig env
 * STUDIO_EMBED_PATH); defaults to /amortized-studio-embed/.
 */
declare const __STUDIO_EMBED_PATH__: string;

const DEFAULT_EMBED_PATH = '/amortized-studio-embed/';

const StudioEmbed: React.FC = () => {
  const embedPath =
    (typeof __STUDIO_EMBED_PATH__ !== 'undefined' && __STUDIO_EMBED_PATH__) || DEFAULT_EMBED_PATH;

  return (
    <iframe
      title="Amortized Studio"
      src={embedPath}
      style={{ flex: 1, width: '100%', height: '100%', border: 0 }}
      allow="clipboard-read; clipboard-write"
    />
  );
};

export default StudioEmbed;
