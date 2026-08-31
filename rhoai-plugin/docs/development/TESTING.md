# Testing

How to run tests, what to test across component boundaries, and a starter set of integration test scenarios for any RHOAI community plugin.

## Running Automated Tests

### Frontend

```bash
npm test                # Run all frontend tests (Jest + jsdom)
npm run test:watch      # Watch mode — re-runs on file changes
npm run test:coverage   # Tests with coverage report
```

To run a single test file:

```bash
npx jest src/app/hooks/useCurrentUser.test.ts
```

### BFF

```bash
cd bff
npm test                # Run BFF tests (Jest + node)
```

### Linting

```bash
npm run lint            # ESLint (src/) + markdownlint (**/*.md)
cd bff && npm run lint  # ESLint (bff/src/)
```

## Contract Testing Principles

Plugins have three independently developed layers — frontend, BFF, and Helm chart — that must agree on shared contracts. Unit tests within a single layer cannot catch mismatches between layers. Contract tests verify these boundaries explicitly.

### What to Test

| Contract | How to Verify |
|---|---|
| **API response shapes** | Assert the same response structure in both the BFF route test (producer) and the frontend hook test (consumer). If either side changes the shape, the other side's test fails. |
| **Label selectors** | Render Helm templates with `helm template` and assert that the labels on Deployments, Services, and Pods match the selectors your application code uses to find those resources. |
| **Status/phase enums** | If the BFF or K8s API returns status values (e.g., `Running`, `Pending`, `Failed`), the frontend must handle every known value plus an `unknown`/default case. Add a test that maps all enum values to UI states. |
| **Auth token forwarding** | Test the full chain: dashboard sends Bearer token → BFF middleware extracts it → BFF uses it to call K8s APIs. A missing or malformed token at any step should produce a clear 401, not a silent failure or HTML error page. |

### Example: API Shape Contract

The BFF route test asserts what the endpoint returns:

```typescript
// bff/src/routes/summary.test.ts
const response = await request(app).get('/summary/my-namespace');
expect(response.body).toEqual(
  expect.objectContaining({
    items: expect.any(Array),
    count: expect.any(Number),
  }),
);
```

The frontend hook test asserts it can consume that shape:

```typescript
// src/app/hooks/useNamespaceSummary.test.ts
mockFetch.mockResolvedValue({
  ok: true,
  json: () => Promise.resolve({ items: [...], count: 3 }),
});
const { result } = renderHook(() => useNamespaceSummary('my-namespace'));
// assert hook exposes items and count correctly
```

If the BFF adds a field or changes a name, both tests must be updated — the failing consumer test catches the mismatch before it reaches production.

## Integration Test Scenarios

Automated tests cover code correctness but cannot verify the full integration with the RHOAI dashboard and a live cluster. The scenarios below are a starter set of manual checks that every plugin should pass before shipping. They surface the kinds of issues that only appear when running against a real environment.

### Plugin Loading

- [ ] Plugin appears in the dashboard sidebar under the correct navigation section
- [ ] Clicking the plugin nav item loads the plugin's first page without errors
- [ ] Browser console shows no Module Federation errors or failed chunk loads

### BFF Proxy

- [ ] BFF endpoints return JSON, not an HTML error page (common when the proxy path is misconfigured)
- [ ] Requests from the frontend reach the BFF service with the correct path prefix stripped
- [ ] BFF responses include appropriate CORS headers if accessed cross-origin

### Authentication

- [ ] Auth token reaches BFF routes (inspect the `Authorization` header in BFF logs or a test endpoint)
- [ ] Requests without a Bearer token return 401, not a 500 or an HTML login page
- [ ] Token expiry or revocation produces a clear error, not a silent data-fetch failure

### RBAC and Permissions

- [ ] K8s API calls respect the authenticated user's RBAC permissions
- [ ] A user without access to a namespace sees an appropriate error, not a blank page
- [ ] Admin-only operations are hidden or disabled for non-admin users

### Error States

- [ ] BFF unreachable: plugin shows a clear error, not a blank page or infinite spinner
- [ ] Invalid or expired token: plugin shows an auth error
- [ ] Missing permissions: plugin shows a permission-denied message
- [ ] Network timeout: plugin recovers gracefully (shows error, allows retry)

### Loading States

- [ ] Pages show skeleton screens while data is loading, not blank pages
- [ ] Skeletons are announced to screen readers (`screenReaderText` on PatternFly Skeleton)
- [ ] After data loads, skeletons are replaced — no flicker or double-render

### Navigation

- [ ] Navigating between plugin pages preserves relevant state (e.g., selected project)
- [ ] Browser back/forward buttons work correctly within the plugin
- [ ] Deep-linking to a plugin page (pasting a URL) loads the correct content
