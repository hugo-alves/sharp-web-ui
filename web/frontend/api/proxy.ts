import type { VercelRequest, VercelResponse } from '@vercel/node';
import { Readable } from 'stream';

const MODAL_API_URL = process.env.MODAL_API_URL || 'https://hugo-alves--sharp-web-ui-fastapi-app.modal.run';
const MODAL_API_KEY = process.env.MODAL_API_KEY;

// Helper to read request body as buffer
async function getRawBody(req: VercelRequest): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of req as unknown as Readable) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }
  return Buffer.concat(chunks);
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  // Extract path from the request URL (remove /api/proxy prefix)
  const urlPath = req.url || '';
  const apiPath = urlPath.replace(/^\/api\/proxy\/?/, '').split('?')[0];

  // Build the target URL
  const targetUrl = `${MODAL_API_URL}/api/${apiPath}`;

  // Forward query params
  const url = new URL(targetUrl);
  Object.entries(req.query).forEach(([key, value]) => {
    if (typeof value === 'string') {
      url.searchParams.set(key, value);
    }
  });

  try {
    // Prepare headers - add API key server-side
    const headers: Record<string, string> = {
      'X-API-Key': MODAL_API_KEY || '',
    };

    // Forward relevant headers
    if (req.headers['content-type']) {
      headers['Content-Type'] = req.headers['content-type'] as string;
    }
    if (req.headers['x-session-id']) {
      headers['X-Session-Id'] = req.headers['x-session-id'] as string;
    }

    // Read raw body for non-GET requests
    let body: Buffer | undefined;
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      body = await getRawBody(req);
    }

    // Make the request to Modal
    const response = await fetch(url.toString(), {
      method: req.method,
      headers,
      body: body ? new Uint8Array(body) : undefined,
    });

    // Forward response headers
    const contentType = response.headers.get('content-type');
    if (contentType) {
      res.setHeader('Content-Type', contentType);
    }

    const contentDisposition = response.headers.get('content-disposition');
    if (contentDisposition) {
      res.setHeader('Content-Disposition', contentDisposition);
    }

    // Handle binary responses (files)
    if (contentType?.includes('application/octet-stream') ||
        contentType?.includes('application/zip') ||
        apiPath.endsWith('.splat')) {
      const buffer = await response.arrayBuffer();
      res.status(response.status).send(Buffer.from(buffer));
    } else {
      // JSON or text response
      const data = await response.text();
      res.status(response.status).send(data);
    }
  } catch (error) {
    console.error('Proxy error:', error);
    res.status(500).json({ error: 'Failed to proxy request to Modal API' });
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
