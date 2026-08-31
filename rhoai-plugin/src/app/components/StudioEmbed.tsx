import React from 'react';
import {
  Button,
  EmptyState,
  EmptyStateBody,
  EmptyStateActions,
  EmptyStateFooter,
} from '@patternfly/react-core';

/**
 * Launcher for the Amortized Studio SPA.
 *
 * Studio runs behind OpenShift oauth-proxy. The OpenShift OAuth server sets
 * `X-Frame-Options: DENY`, so the sign-in redirect cannot render inside the
 * dashboard iframe (cross-origin). We therefore open the studio top-level in a
 * new tab, where the OAuth/SSO flow works normally.
 *
 * (A future same-origin embed — serving studio through the dashboard proxy with
 * `authorize: true`, like the mlflowEmbedded plugin — would restore an inline
 * iframe, but requires the studio SPA to support a base path.)
 *
 * URL resolution: build-time __STUDIO_URL__ (BuildConfig env STUDIO_URL) →
 * window.__AMORTIZED_STUDIO_URL__ → default.
 */
declare global {
  interface Window {
    __AMORTIZED_STUDIO_URL__?: string;
  }
}
declare const __STUDIO_URL__: string;

const DEFAULT_STUDIO_URL =
  'https://amortized-studio-meyceoz-amortized.apps.rosa.h0p8o2c2b7w3r1y.fjws.p3.openshiftapps.com/';

const StudioEmbed: React.FC = () => {
  const studioUrl =
    (typeof __STUDIO_URL__ !== 'undefined' && __STUDIO_URL__) ||
    (typeof window !== 'undefined' && window.__AMORTIZED_STUDIO_URL__) ||
    DEFAULT_STUDIO_URL;

  return (
    <EmptyState titleText="Amortized Studio" headingLevel="h1">
      <EmptyStateBody>
        Build task-specific fine-tuned models — datasets, training jobs, models,
        and the Morty assistant. Studio opens in a new tab so you can sign in
        securely with your OpenShift account. Your workspace is provisioned on
        first launch and isolated to you.
      </EmptyStateBody>
      <EmptyStateFooter>
        <EmptyStateActions>
          <Button
            variant="primary"
            component="a"
            href={studioUrl}
            target="_blank"
            rel="noopener noreferrer"
          >
            Launch Amortized Studio
          </Button>
        </EmptyStateActions>
      </EmptyStateFooter>
    </EmptyState>
  );
};

export default StudioEmbed;
