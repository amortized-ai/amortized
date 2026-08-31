// Identity-aware studio gateway (hybrid model).
//
// Sits behind oauth-proxy, which authenticates the OpenShift user and passes
// their identity via X-Forwarded-User. For each user the gateway:
//   - provisions an isolated backend stack on demand (server + Postgres + ns)
//   - routes /api and /agent to *that user's* server
//   - routes /mlflow to a shared MLflow upstream
//   - serves the shared studio SPA (static) for everything else

const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const { ensureUserStack, getState, markForRetry } = require('./provision');
const { renderSplash } = require('./splash');
const { resolveUser, identityDebug } = require('./auth');

const PORT = parseInt(process.env.PORT || '8080', 10);
// Studio SPA static files (built with base path = EMBED_BASE_PATH) are baked
// into the image and served directly by the gateway.
const STUDIO_DIST = process.env.STUDIO_DIST || path.join(__dirname, '..', 'studio-dist');
const MLFLOW_UPSTREAM = process.env.MLFLOW_UPSTREAM || '';

const app = express();

// --- Embed base-path handling -----------------------------------------------
// When served through the RHOAI dashboard proxy, requests may arrive under a
// base prefix (e.g. /amortized-studio-embed/...). Strip it if present so all
// downstream routing works at the root. Tolerant of either behavior (dashboard
// preserving or rewriting the prefix).
const EMBED_BASE = (process.env.EMBED_BASE_PATH || '').replace(/\/+$/, '');
app.use((req, _res, next) => {
  if (EMBED_BASE && (req.url === EMBED_BASE || req.url.startsWith(`${EMBED_BASE}/`))) {
    req.url = req.url.slice(EMBED_BASE.length) || '/';
  }
  next();
});

// --- Identity ---------------------------------------------------------------
// Resolve the acting user once per request (proxy header or TokenReview) and
// stash it; downstream handlers read it synchronously.
app.use((req, _res, next) => {
  resolveUser(req)
    .then((u) => { req.amortizedUser = u || ''; next(); })
    .catch(() => { req.amortizedUser = ''; next(); });
});

function currentUser(req) {
  return req.amortizedUser || '';
}

// Debug: what identity did the gateway see/resolve for this request?
app.get('/gateway/whoami', (req, res) => {
  res.json({ user: currentUser(req) || null, ...identityDebug(req) });
});

app.get('/gateway/healthz', (_req, res) => res.json({ ok: true }));

// Status/bootstrap endpoint the splash polls.
app.get('/gateway/ready', (req, res) => {
  const user = currentUser(req);
  if (!user) return res.status(401).json({ error: 'no authenticated user' });
  const state = ensureUserStack(user); // idempotent: kicks off provisioning
  res.json({ user, ...state });
});

// Clear a failed provisioning attempt and try again.
app.post('/gateway/retry', (req, res) => {
  const user = currentUser(req);
  if (!user) return res.status(401).json({ error: 'no authenticated user' });
  markForRetry(user);
  const state = ensureUserStack(user);
  res.json({ user, ...state });
});

// --- Per-user backend proxy (/api, /agent, /mcp) ----------------------------
// Mounted at root with a pathFilter so the FULL path (e.g. /api/v1/health) is
// preserved to the upstream. Target is computed per-request from the user's ns.
const BACKEND_PREFIXES = ['/api', '/agent', '/mcp'];
const isBackendPath = (p) => BACKEND_PREFIXES.some((b) => p === b || p.startsWith(`${b}/`));

// Gate backend traffic on readiness; trigger provisioning if needed.
app.use((req, res, next) => {
  if (!isBackendPath(req.path)) return next();
  const user = currentUser(req);
  if (!user) return res.status(401).json({ error: 'no authenticated user' });
  const state = ensureUserStack(user);
  if (state.state === 'ready') return next();
  const code = state.state === 'error' ? 500 : 503;
  res.status(code).json({ status: state.state, detail: state.error || 'Provisioning your environment…' });
});

app.use(createProxyMiddleware({
  pathFilter: ['/api/**', '/agent/**', '/mcp/**'],
  changeOrigin: true,
  ws: true,
  router: (req) => `http://amortized-server.${getState(currentUser(req)).ns}.svc.cluster.local:8000`,
  on: {
    error: (_err, _req, res) => {
      if (!res || res.writableEnded) return;
      res.writeHead(503, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'starting', detail: 'Backend is starting up. Retry shortly.' }));
    },
  },
}));

// --- Shared MLflow ----------------------------------------------------------
if (MLFLOW_UPSTREAM) {
  app.use(createProxyMiddleware({ pathFilter: ['/mlflow/**'], target: MLFLOW_UPSTREAM, changeOrigin: true }));
} else {
  app.use('/mlflow', (_req, res) => res.status(503).json({ status: 'unconfigured', detail: 'Shared MLflow not configured yet.' }));
}

// --- Provisioning splash for the initial navigation -------------------------
// For HTML navigations (not assets/api), show a splash until the user's backend
// is ready; the splash polls /gateway/ready and reloads into the SPA.
const isAsset = (p) => /\.[a-z0-9]+$/i.test(p);
app.use((req, res, next) => {
  if (req.method !== 'GET') return next();
  if (isBackendPath(req.path) || req.path.startsWith('/mlflow') || req.path.startsWith('/gateway')) return next();
  if (isAsset(req.path) || !req.accepts('html')) return next();
  const user = currentUser(req);
  if (!user) return next();
  const state = ensureUserStack(user);
  if (state.state === 'ready') return next();
  res.set('Cache-Control', 'no-store');
  return res.status(200).send(renderSplash(state));
});

// --- Shared studio SPA (static, catch-all) ----------------------------------
// Assets are served directly; unmatched navigations fall back to index.html
// for client-side routing.
app.use(express.static(STUDIO_DIST, { index: false }));
app.get('*', (req, res, next) => {
  if (isBackendPath(req.path) || req.path.startsWith('/mlflow') || req.path.startsWith('/gateway')) {
    return next();
  }
  res.sendFile(path.join(STUDIO_DIST, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`studio-gateway listening on :${PORT}`);
  console.log(`  static  -> ${STUDIO_DIST}`);
  console.log(`  base    -> ${EMBED_BASE || '(root)'}`);
  console.log(`  mlflow  -> ${MLFLOW_UPSTREAM || '(unconfigured)'}`);
});
