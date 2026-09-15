import express from 'express';
import { namespaceSummaryHandler } from './routes/namespaceSummary';
import { getK8sBaseUrl } from './utils/k8sClient';

const app = express();
const PORT = parseInt(process.env.PORT || '3000', 10);
const POD_NAMESPACE = process.env.POD_NAMESPACE ?? 'cp-amortized-studio';

app.use((req, res, next) => {
  const start = Date.now();
  res.on('finish', () => {
    console.log(`${req.method} ${req.originalUrl} ${res.statusCode} ${Date.now() - start}ms`);
  });
  next();
});

app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));

app.get('/api/config', (_req, res) => {
  res.json({ bffNamespace: POD_NAMESPACE });
});

app.get('/api/namespace-summary', namespaceSummaryHandler);

app.listen(PORT, () => {
  try {
    const baseUrl = getK8sBaseUrl();
    console.log(`BFF listening on port ${PORT}`);
    console.log(`K8s API target: ${baseUrl}`);
  } catch {
    console.error(`BFF listening on port ${PORT}`);
    console.error('WARNING: K8s API is not configured. Set K8S_API_BASE or run in-cluster.');
    console.error('  Example: K8S_API_BASE=$(oc whoami --show-server) npm run start:dev');
    console.error('  All API requests will fail with 502 until this is set.');
  }
});

export default app;
