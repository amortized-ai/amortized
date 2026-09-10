import https from 'https';
import { EventEmitter } from 'events';
import { k8sRequest, getK8sBaseUrl } from '../src/utils/k8sClient';

jest.mock('https');

const mockedHttps = jest.mocked(https);

function createMockResponse(statusCode: number, body: string) {
  const res = new EventEmitter() as any;
  res.statusCode = statusCode;
  process.nextTick(() => {
    res.emit('data', body);
    res.emit('end');
  });
  return res;
}

describe('getK8sBaseUrl', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    process.env = { ...originalEnv };
    delete process.env.K8S_API_BASE;
    delete process.env.KUBERNETES_SERVICE_HOST;
    delete process.env.KUBERNETES_SERVICE_PORT;
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('returns K8S_API_BASE when set', () => {
    process.env.K8S_API_BASE = 'https://my-cluster:6443';
    expect(getK8sBaseUrl()).toBe('https://my-cluster:6443');
  });

  it('constructs URL from KUBERNETES_SERVICE_HOST and PORT', () => {
    process.env.KUBERNETES_SERVICE_HOST = '10.0.0.1';
    process.env.KUBERNETES_SERVICE_PORT = '443';
    expect(getK8sBaseUrl()).toBe('https://10.0.0.1:443');
  });

  it('throws when no env vars are set', () => {
    expect(() => getK8sBaseUrl()).toThrow('K8s API not configured');
  });
});

describe('k8sRequest', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    jest.resetAllMocks();
    process.env = { ...originalEnv };
    process.env.K8S_API_BASE = 'https://my-cluster:6443';
  });

  afterAll(() => {
    process.env = originalEnv;
  });

  it('sends Authorization header with the token', async () => {
    const mockReq = new EventEmitter() as any;
    mockReq.end = jest.fn();

    mockedHttps.request.mockImplementation((_opts: any, callback: any) => {
      callback(createMockResponse(200, '{"items":[]}'));
      return mockReq;
    });

    await k8sRequest('test-token', '/api/v1/pods');

    const callArgs = mockedHttps.request.mock.calls[0][0] as any;
    expect(callArgs.headers.Authorization).toBe('Bearer test-token');
  });

  it('constructs the correct URL path', async () => {
    const mockReq = new EventEmitter() as any;
    mockReq.end = jest.fn();

    mockedHttps.request.mockImplementation((_opts: any, callback: any) => {
      callback(createMockResponse(200, '{}'));
      return mockReq;
    });

    await k8sRequest('token', '/apis/project.openshift.io/v1/projects');

    const callArgs = mockedHttps.request.mock.calls[0][0] as any;
    expect(callArgs.hostname).toBe('my-cluster');
    expect(callArgs.port).toBe('6443');
    expect(callArgs.path).toBe('/apis/project.openshift.io/v1/projects');
  });

  it('rejects on non-2xx responses', async () => {
    const mockReq = new EventEmitter() as any;
    mockReq.end = jest.fn();

    mockedHttps.request.mockImplementation((_opts: any, callback: any) => {
      callback(createMockResponse(403, '{"message":"forbidden"}'));
      return mockReq;
    });

    await expect(k8sRequest('bad-token', '/api/v1/pods')).rejects.toThrow(
      'K8s API returned 403',
    );
  });

  it('rejects on request error', async () => {
    const mockReq = new EventEmitter() as any;
    mockReq.end = jest.fn();

    mockedHttps.request.mockImplementation(() => {
      process.nextTick(() => mockReq.emit('error', new Error('ECONNREFUSED')));
      return mockReq;
    });

    await expect(k8sRequest('token', '/api/v1/pods')).rejects.toThrow(
      'ECONNREFUSED',
    );
  });
});
