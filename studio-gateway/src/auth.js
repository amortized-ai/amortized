// User identity resolution.
//
// Two deployment modes:
//   - Behind oauth-proxy: identity arrives as X-Forwarded-User (+ variants).
//   - Behind the RHOAI dashboard proxy (authorize:true): the dashboard forwards
//     the user's OpenShift token; we resolve the username via a TokenReview
//     (the gateway SA holds system:auth-delegator).
//
// Results are cached briefly by token to avoid a TokenReview per request.

const k8s = require('@kubernetes/client-node');

const kc = new k8s.KubeConfig();
kc.loadFromCluster();
const authApi = kc.makeApiClient(k8s.AuthenticationV1Api);

const DEV_USER = process.env.DEV_USER || '';
const CACHE_TTL_MS = 60_000;
const tokenCache = new Map(); // token -> { user, exp }

function headerUser(req) {
  return (
    req.headers['x-forwarded-preferred-username'] ||
    req.headers['x-forwarded-user'] ||
    req.headers['x-forwarded-email'] ||
    ''
  );
}

function bearerToken(req) {
  const auth = req.headers['authorization'] || '';
  if (auth.toLowerCase().startsWith('bearer ')) return auth.slice(7);
  return req.headers['x-forwarded-access-token'] || '';
}

async function usernameFromToken(token) {
  const now = Date.now();
  const hit = tokenCache.get(token);
  if (hit && hit.exp > now) return hit.user;
  try {
    const res = await authApi.createTokenReview({ body: { spec: { token } } });
    const status = res.status || res.body?.status || {};
    const user = status.authenticated ? status.user?.username || '' : '';
    tokenCache.set(token, { user, exp: now + CACHE_TTL_MS });
    return user;
  } catch (err) {
    console.error('TokenReview failed:', err?.body || err?.message || err);
    return '';
  }
}

// Resolve the acting user for a request (async). Prefer proxy headers; fall back
// to token review; finally DEV_USER (local/testing only).
async function resolveUser(req) {
  const h = headerUser(req);
  if (h) return h;
  const token = bearerToken(req);
  if (token) {
    const u = await usernameFromToken(token);
    if (u) return u;
  }
  return DEV_USER;
}

// Debug helper: which identity signals are present on the request.
function identityDebug(req) {
  return {
    headerUser: headerUser(req) || null,
    hasBearer: !!(req.headers['authorization'] || '').toLowerCase().startsWith('bearer '),
    hasForwardedAccessToken: !!req.headers['x-forwarded-access-token'],
    forwardedHeaders: Object.keys(req.headers).filter((k) => k.startsWith('x-forwarded')),
  };
}

module.exports = { resolveUser, identityDebug };
