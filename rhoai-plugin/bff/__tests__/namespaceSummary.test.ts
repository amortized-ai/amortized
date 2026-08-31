import { Request, Response } from 'express';
import { namespaceSummaryHandler } from '../src/routes/namespaceSummary';
import { k8sRequest } from '../src/utils/k8sClient';

jest.mock('../src/utils/k8sClient');

const mockedK8sRequest = jest.mocked(k8sRequest);

function createMockReqRes(headers: Record<string, string> = {}) {
  const req = { headers } as unknown as Request;
  const res = {
    status: jest.fn().mockReturnThis(),
    json: jest.fn().mockReturnThis(),
  } as unknown as Response;
  return { req, res };
}

describe('namespaceSummaryHandler', () => {
  beforeEach(() => {
    jest.resetAllMocks();
  });

  it('returns 401 when no Authorization header', async () => {
    const { req, res } = createMockReqRes();

    await namespaceSummaryHandler(req, res);

    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalledWith({
      error: 'Missing or invalid Authorization header',
    });
  });

  it('returns 401 when Authorization header is not Bearer', async () => {
    const { req, res } = createMockReqRes({
      authorization: 'Basic abc123',
    });

    await namespaceSummaryHandler(req, res);

    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('returns aggregated namespace data on success', async () => {
    const { req, res } = createMockReqRes({
      authorization: 'Bearer valid-token',
    });

    mockedK8sRequest.mockImplementation((_token, path) => {
      if (path.includes('/projects')) {
        return Promise.resolve({
          items: [
            { metadata: { name: 'ns-a' }, status: { phase: 'Active' } },
            { metadata: { name: 'ns-b' }, status: { phase: 'Active' } },
          ],
        });
      }
      if (path.includes('ns-a/pods')) {
        return Promise.resolve({
          items: [
            { status: { phase: 'Running' } },
            { status: { phase: 'Running' } },
            { status: { phase: 'Pending' } },
          ],
        });
      }
      if (path.includes('ns-b/pods')) {
        return Promise.resolve({
          items: [
            { status: { phase: 'Failed' } },
            { status: { phase: 'Succeeded' } },
          ],
        });
      }
      return Promise.resolve({ items: [] });
    });

    await namespaceSummaryHandler(req, res);

    expect(res.json).toHaveBeenCalledWith({
      namespaces: [
        {
          name: 'ns-a',
          phase: 'Active',
          pods: {
            total: 3,
            running: 2,
            pending: 1,
            succeeded: 0,
            failed: 0,
            unknown: 0,
          },
        },
        {
          name: 'ns-b',
          phase: 'Active',
          pods: {
            total: 2,
            running: 0,
            pending: 0,
            succeeded: 1,
            failed: 1,
            unknown: 0,
          },
        },
      ],
      errors: [],
    });
  });

  it('handles partial failures gracefully', async () => {
    const { req, res } = createMockReqRes({
      authorization: 'Bearer valid-token',
    });

    mockedK8sRequest.mockImplementation((_token, path) => {
      if (path.includes('/projects')) {
        return Promise.resolve({
          items: [
            { metadata: { name: 'ns-ok' }, status: { phase: 'Active' } },
            { metadata: { name: 'ns-fail' }, status: { phase: 'Active' } },
          ],
        });
      }
      if (path.includes('ns-ok/pods')) {
        return Promise.resolve({
          items: [{ status: { phase: 'Running' } }],
        });
      }
      if (path.includes('ns-fail/pods')) {
        return Promise.reject(new Error('forbidden'));
      }
      return Promise.resolve({ items: [] });
    });

    await namespaceSummaryHandler(req, res);

    const response = (res.json as jest.Mock).mock.calls[0][0];
    expect(response.namespaces).toHaveLength(1);
    expect(response.namespaces[0].name).toBe('ns-ok');
    expect(response.errors).toEqual([
      { name: 'ns-fail', error: 'forbidden' },
    ]);
  });

  it('returns 502 when projects call fails', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    const { req, res } = createMockReqRes({
      authorization: 'Bearer valid-token',
    });

    mockedK8sRequest.mockRejectedValue(new Error('connection refused'));

    await namespaceSummaryHandler(req, res);

    expect(res.status).toHaveBeenCalledWith(502);
    expect(res.json).toHaveBeenCalledWith({ error: 'connection refused' });
    consoleSpy.mockRestore();
  });
});
