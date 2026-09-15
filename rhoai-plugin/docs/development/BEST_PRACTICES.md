# Development Best Practices

Common pitfalls and development patterns for RHOAI community plugins, distilled from real plugin development experience. Each section covers a category of bugs that are easy to introduce and hard to catch without integration testing.

## BFF Endpoint Development

### Auth Guards

Use runtime checks with proper HTTP status codes. Never use TypeScript non-null assertions (`!`) on authentication values — they suppress compile-time warnings but crash at runtime when the token is missing.

```typescript
// Bad — crashes with unhelpful error if token is undefined
const token = req.token!;

// Good — returns a clear 401 response
const token = req.token;
if (!token) {
  return res.status(401).json({ message: 'Authentication required' });
}
```

### Input Validation

Validate all query and body parameters at route boundaries:

- **Port ranges**: Reject values outside 1–65535.
- **String character sets**: Validate that URIs, namespace names, and label values conform to their expected character sets before passing them to K8s APIs.
- **Empty-array edge cases**: `Array.every()` returns `true` for an empty array (vacuous truth). If you use it for RBAC permission checks, an empty permission list bypasses the check entirely.

```typescript
// Bad — passes if requiredPermissions is empty
const hasAccess = requiredPermissions.every((p) => userPerms.includes(p));

// Good — explicit empty check
const hasAccess =
  requiredPermissions.length > 0 &&
  requiredPermissions.every((p) => userPerms.includes(p));
```

### Error Handling

- Always `.catch()` promises. Unhandled promise rejections crash the process in newer Node.js versions.
- Never use `|| true` or similar patterns to swallow errors — they hide real failures.
- Sanitize Kubernetes API error bodies before returning them to clients. K8s error objects can contain internal cluster details that should not be exposed.
- Map errors to appropriate HTTP status codes: 401 for auth failures, 403 for RBAC denials, 404 for missing resources, 502 for upstream service failures.

### Pagination

Return pagination metadata from list endpoints from the start:

```typescript
// Good — pagination-ready from day one
return res.json({
  items: results,
  count: total,
  page: requestedPage,
  pageSize: requestedPageSize,
});
```

Retrofitting pagination into an endpoint that returns a bare array is a breaking change for every consumer.

### HTTP Client Type Safety

Not every HTTP response has a JSON body. Handle these cases explicitly:

- **204 No Content**: Check the status code before calling `.json()`.
- **Plain-text health endpoints**: Some services return `"OK"` instead of `{"status": "ok"}`. Use `response.text()` and check the content type.
- **Empty bodies**: `response.json()` throws on an empty string. Guard with a length or content-type check.

```typescript
// Bad — crashes on 204 or plain-text responses
const data = (await response.json()) as MyType;

// Good — handles non-JSON responses
if (response.status === 204) {
  return null;
}
const contentType = response.headers.get('content-type') ?? '';
if (!contentType.includes('application/json')) {
  const text = await response.text();
  return { raw: text };
}
const data = (await response.json()) as MyType;
```

## React Hooks and Components

### Cleanup with Refs

Hook cleanup functions run after re-renders. If the cleanup captures a variable from the render closure, it uses the stale value. Use a ref to ensure cleanup always acts on the current value.

```typescript
// Bad — cleanup captures stale controller from the render that created it
useEffect(() => {
  const controller = new AbortController();
  fetchData(controller.signal);
  return () => controller.abort();
}, [dependency]);

// Good — ref always points to the latest controller
const controllerRef = useRef<AbortController>();
useEffect(() => {
  controllerRef.current?.abort();
  controllerRef.current = new AbortController();
  fetchData(controllerRef.current.signal);
  return () => controllerRef.current?.abort();
}, [dependency]);
```

### Error Boundary Keying

Key error boundaries by navigation parameters. Without a key, navigating from one errored item to another keeps the error boundary in its error state instead of resetting.

```tsx
<ErrorBoundary key={itemId}>
  <ItemDetail id={itemId} />
</ErrorBoundary>
```

### Accessibility

- **Narrow `aria-live` regions**: Wrapping an entire `PageSection` with `aria-live="polite"` causes screen readers to re-announce the full DOM subtree on every status change. Wrap only the text that actually changes.
- **Skeleton screen readers**: Add `screenReaderText` to PatternFly `Skeleton` components so assistive technology users know content is loading.

### PatternFly Conventions

- Use PatternFly utility classes (`pf-v6-u-mt-md`) instead of inline styles for spacing, alignment, and sizing.
- Use accurate action labels that describe what happened (e.g., "Copied to clipboard"), not labels describing when the button appears.

## Helm Chart Templates

### Type Coercion

Helm values are strings by default. Numeric values in templates need the `| int` filter, or comparisons and arithmetic will behave unexpectedly.

```yaml
# Bad — port is a string, may cause issues in some K8s fields
containerPort: {{ .Values.service.port }}

# Good — explicitly cast to integer
containerPort: {{ .Values.service.port | int }}
```

### Name Truncation

Helm's `trunc 63` idiom limits names to 63 characters (K8s label value limit). Test with maximum-length release names and verify that suffixes like `-postgres` or `-bff` still fit after truncation.

### Resource Policies

Add `helm.sh/resource-policy: keep` to PersistentVolumeClaims and Secrets that contain generated credentials. Without this annotation, `helm uninstall` deletes them and the data is lost.

```yaml
metadata:
  annotations:
    "helm.sh/resource-policy": keep
```

### Value Validation

Document constraints directly in `values.yaml` comments: allowed character sets, valid ranges, maximum lengths. This is the first place developers look when configuring a chart.

### Label Consistency

Label selectors in application code (e.g., when listing pods or services belonging to a deployment) must exactly match the labels rendered by Helm templates. A mismatch means the application cannot find its own resources. Test selectors against actual rendered output using `helm template`.

## Security

### Path Validation

Use `startsWith()` on resolved (absolute) paths, never `includes()`. A substring match is trivially bypassed with path traversal.

```typescript
// Bad — '../etc/passwd' includes '/allowed/path' if crafted correctly
if (requestedPath.includes(allowedBase)) { /* ... */ }

// Good — checks that the resolved path begins with the allowed directory
const resolved = path.resolve(requestedPath);
if (!resolved.startsWith(path.resolve(allowedBase))) {
  return res.status(403).json({ message: 'Access denied' });
}
```

### CORS

If origin checks validate a port number, reject port 0 and ports above 65535. These are invalid but may pass naive range checks.

### Error Exposure

Never forward raw internal error objects to HTTP clients. Extract only the user-facing message.

```typescript
// Bad — leaks stack traces, internal paths, K8s API details
res.status(500).json(error);

// Good — controlled error response
res.status(500).json({
  message: error instanceof Error ? error.message : 'Internal server error',
});
```

### Auth Coverage

Verify that every non-health route has authentication middleware. The simplest test: send a request without a `Bearer` token and confirm a 401 response.

## Cross-Component Contract Testing

These tests verify that independently developed components (frontend hooks, BFF routes, Helm templates) agree on their shared contracts.

| What to Test | Producer | Consumer |
|---|---|---|
| API response shapes | BFF route test asserts the response structure | Frontend hook test asserts it can parse the structure |
| Label selectors | Helm template renders labels on resources | Application code uses the same labels to find resources |
| Status/phase enums | BFF or K8s API returns a set of status values | Frontend handles every value (including unknown ones) |
| Auth token forwarding | Dashboard forwards Bearer token to BFF | BFF middleware extracts and validates the token |

A mismatch in any of these contracts causes failures that only surface during integration testing.

## Pre-PR Checklist

A condensed checklist to review before opening a pull request. Not every item applies to every PR — scan for relevance.

### BFF

- [ ] Every non-health route has auth middleware
- [ ] Query/body params are validated at the route boundary
- [ ] Promises have `.catch()` handlers
- [ ] K8s error bodies are sanitized before returning to clients
- [ ] List endpoints return `{ items, count, page, pageSize }`
- [ ] Non-JSON responses (204, plain text) are handled explicitly

### Frontend

- [ ] `useEffect` cleanup uses refs, not closure-captured values
- [ ] Error boundaries are keyed by navigation parameters
- [ ] `aria-live` regions wrap only the changing text
- [ ] Skeletons have `screenReaderText`
- [ ] PatternFly utility classes used instead of inline styles

### Helm

- [ ] Numeric values use `| int` in templates
- [ ] Name truncation tested with max-length release names
- [ ] PVCs and credential Secrets have `resource-policy: keep`
- [ ] Value constraints documented in `values.yaml` comments

### Security

- [ ] Path validation uses `startsWith()` on resolved paths
- [ ] CORS port checks reject 0 and >65535
- [ ] Error responses contain only the message, not raw error objects
- [ ] Missing Bearer token returns 401

### Cross-Component

- [ ] API response shapes match between BFF tests and frontend tests
- [ ] Label selectors match rendered Helm template labels
- [ ] Status/phase enums handled exhaustively in the frontend
- [ ] Auth token forwarding tested end-to-end
