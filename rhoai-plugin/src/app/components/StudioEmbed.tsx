import React from 'react';

/**
 * Full-bleed embed of the Amortized Studio SPA.
 *
 * The studio is a complete, self-contained app (Vite/React) served behind its
 * own nginx on OpenShift. Rather than re-implement its UI as native PatternFly
 * federated modules, this plugin embeds the running studio the same way the
 * apache-superset / mlflowEmbedded community plugins embed their apps.
 *
 * Resolution order for the studio URL:
 *   1. window.__AMORTIZED_STUDIO_URL__  — injected at runtime by the plugin's
 *      nginx (config.js) or, later, resolved per-project by the plugin BFF
 *      (on-demand deployment, the superset "Instance Management" pattern).
 *   2. STUDIO_URL build-time default    — the shared experimental deployment.
 */
declare global {
  interface Window {
    __AMORTIZED_STUDIO_URL__?: string;
  }
}
// Injected at build time via webpack DefinePlugin (BuildConfig env STUDIO_URL).
declare const __STUDIO_URL__: string;

const DEFAULT_STUDIO_URL =
  'https://amortized-studio-meyceoz-amortized.apps.rosa.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com/';

const StudioEmbed: React.FC = () => {
  const studioUrl =
    (typeof __STUDIO_URL__ !== 'undefined' && __STUDIO_URL__) ||
    (typeof window !== 'undefined' && window.__AMORTIZED_STUDIO_URL__) ||
    DEFAULT_STUDIO_URL;

  return (
    <iframe
      title="Amortized Studio"
      src={studioUrl}
      style={{ flex: 1, width: '100%', height: '100%', border: 0 }}
      allow="clipboard-read; clipboard-write"
    />
  );
};

export default StudioEmbed;
