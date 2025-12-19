import axios from 'axios';

// Always use the proxy path - vite.config.ts proxies to Vercel in dev
const API_BASE_URL = '/api/proxy';

// Generate or retrieve session ID for privacy isolation
function getSessionId(): string {
  const STORAGE_KEY = 'sharp_session_id';
  let sessionId = localStorage.getItem(STORAGE_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(STORAGE_KEY, sessionId);
  }
  return sessionId;
}

export const SESSION_ID = getSessionId();

const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add session ID to all requests
api.interceptors.request.use((config) => {
  config.headers['X-Session-Id'] = SESSION_ID;
  return config;
});

export interface Job {
  id: string;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  error?: string;
  result?: {
    num_gaussians: number;
    image_width: number;
    image_height: number;
    focal_length_px: number;
    output_path: string;
  };
}

export interface PreviewData {
  positions: number[];
  colors: number[];
  num_points: number;
  total_gaussians: number;
}

export async function uploadImages(files: File[]): Promise<Job[]> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await api.post('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data.jobs;
}

export async function getJobStatus(jobId: string): Promise<Job> {
  const response = await api.get(`/status/${jobId}`);
  return response.data;
}

export async function listJobs(): Promise<Job[]> {
  const response = await api.get('/jobs');
  return response.data.jobs;
}

export async function getPreviewData(jobId: string): Promise<PreviewData> {
  const response = await api.get(`/preview/${jobId}`);
  return response.data;
}

// Helper to get full URL with session for direct browser requests
export function getAuthenticatedUrl(path: string): string {
  const baseUrl = `${API_BASE_URL}${path}`;
  const params = new URLSearchParams();
  params.set('session_id', SESSION_ID);
  return `${baseUrl}?${params.toString()}`;
}

export function getDownloadUrl(jobId: string): string {
  return getAuthenticatedUrl(`/download/${jobId}`);
}

export function getDownloadAllUrl(): string {
  return getAuthenticatedUrl('/download-all');
}

export function getThumbnailUrl(jobId: string): string {
  return getAuthenticatedUrl(`/thumbnail/${jobId}`);
}

export function getSplatUrl(jobId: string): string {
  return getAuthenticatedUrl(`/splat/${jobId}.splat`);
}

export async function deleteJob(jobId: string): Promise<void> {
  await api.delete(`/jobs/${jobId}`);
}

export async function clearAllJobs(): Promise<void> {
  await api.delete('/jobs');
}

export default api;
