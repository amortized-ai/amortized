// Identity-aware studio gateway (hybrid model).
//
// Sits behind oauth-proxy, which authenticates the OpenShift user and passes
// their identity via X-Forwarded-User. For each user the gateway:
//   - provisions an isolated backend stack on demand (server + Postgres + ns)
//   - routes /api and /agent to *that user's* server
//   - routes /mlflow to a shared MLflow upstream
//   - serves the shared studio SPA (static) for everything else

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const { ensureUserStack, getState } = require('./provision');

const PORT = parseInt(process.env.PORT || '8080', 10);
const STUDIO_STATIC_UPSTREAM = process.env.STUDIO_STATIC_UPSTREAM || 'http://amortized-studio-static:8080';
const MLFLOW_UPSTREAM = process.env.MLFLOW_UPSTREAM || '';
// Dev/testing override: assume this user when no oauth-proxy header is present.
const DEV_USER = process.env.DEV_USER || '';

const app = express();

// --- Identity ---------------------------------------------------------------
function currentUser(req) {
  return (
    req.headers['x-forwarded-preferred-username'] ||
    req.headers['x-forwarded-user'] ||
    req.headers['x-forwarded-email'] ||
    DEV_USER ||
    ''
  );
}

app.get('/gateway/healthz', (_req, res) => res.json({ ok: true }));

// Status/bootstrap endpoint the SPA (or a splash) can poll.
app.get('/gateway/ready', (req, res) => {
  const user = currentUser(req);
  if (!user) return res.status(401).json({ error: 'no authenticated user' });
  const state = ensureUserStack(user); // idempotent: kicks off provisioning
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

// --- Shared studio SPA (static, catch-all) ----------------------------------
app.use(createProxyMiddleware({ target: STUDIO_STATIC_UPSTREAM, changeOrigin: true }));

app.listen(PORT, () => {
  console.log(`studio-gateway listening on :${PORT}`);
  console.log(`  static  -> ${STUDIO_STATIC_UPSTREAM}`);
  console.log(`  mlflow  -> ${MLFLOW_UPSTREAM || '(unconfigured)'}`);
});
