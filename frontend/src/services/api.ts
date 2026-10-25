import type {
  PredictResponse,
  CommentsListResponse,
  CommentRecord,
  ModelInfo,
  HealthResponse,
} from '../types/api';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error(`Failed to load models: ${res.statusText}`);
  return res.json();
}

export async function postPredict(text: string): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchComments(
  page = 1,
  limit = 10,
  action?: string | null
): Promise<CommentsListResponse> {
  const params = new URLSearchParams({
    page: String(page),
    limit: String(limit),
  });
  if (action && action !== 'all') {
    params.append('action', action);
  }
  const res = await fetch(`${API_BASE}/comments?${params.toString()}`);
  if (!res.ok) throw new Error(`Failed to fetch comments: ${res.statusText}`);
  return res.json();
}

export async function fetchCommentById(id: string): Promise<CommentRecord> {
  const res = await fetch(`${API_BASE}/comments/${id}`);
  if (!res.ok) throw new Error(`Comment not found: ${res.statusText}`);
  return res.json();
}

// Research endpoints
export async function fetchResearchOverview(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/overview`);
  if (!res.ok) throw new Error(`Failed to load research overview: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchEDA(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/eda`);
  if (!res.ok) throw new Error(`Failed to load EDA data: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchSplits(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/splits`);
  if (!res.ok) throw new Error(`Failed to load splits data: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchModels(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/models`);
  if (!res.ok) throw new Error(`Failed to load models data: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchResults(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/results`);
  if (!res.ok) throw new Error(`Failed to load results: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchComparisons(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/comparisons`);
  if (!res.ok) throw new Error(`Failed to load comparisons: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchImbalance(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/imbalance`);
  if (!res.ok) throw new Error(`Failed to load imbalance data: ${res.statusText}`);
  return res.json();
}

export async function fetchResearchLimitations(): Promise<any> {
  const res = await fetch(`${API_BASE}/research/limitations`);
  if (!res.ok) throw new Error(`Failed to load limitations: ${res.statusText}`);
  return res.json();
}
