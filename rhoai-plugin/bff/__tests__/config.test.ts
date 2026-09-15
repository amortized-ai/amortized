import http from 'http';
import express from 'express';

const TEST_POD_NAMESPACE = 'custom-ns';

function request(port: number, path: string): Promise<{ statusCode: number; body: string }> {
  return new Promise((resolve, reject) => {
    http.get(`http://127.0.0.1:${port}${path}`, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => resolve({ statusCode: res.statusCode!, body }));
    }).on('error', reject);
  });
}

describe('/api/config endpoint', () => {
  let server: http.Server;
  let port: number;

  beforeAll((done) => {
    const originalEnv = process.env.POD_NAMESPACE;
    process.env.POD_NAMESPACE = TEST_POD_NAMESPACE;

    jest.isolateModules(() => {
      const app = express();
      const POD_NAMESPACE = process.env.POD_NAMESPACE ?? 'cp-amortized-studio';

      app.get('/api/config', (_req, res) => {
        res.json({ bffNamespace: POD_NAMESPACE });
      });

      server = app.listen(0, () => {
        port = (server.address() as any).port;
        process.env.POD_NAMESPACE = originalEnv;
        done();
      });
    });
  });

  afterAll((done) => {
    server.close(done);
  });

  it('returns the BFF namespace', async () => {
    const res = await request(port, '/api/config');
    expect(res.statusCode).toBe(200);
    expect(JSON.parse(res.body)).toEqual({ bffNamespace: TEST_POD_NAMESPACE });
  });
});
