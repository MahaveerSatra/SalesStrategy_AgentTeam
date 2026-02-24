/**
 * API client for the Sales Research Agent backend.
 */

import type {
  ResearchRequest,
  ResearchResponse,
  ResearchState,
  ThreadSummary,
  NodeTrace,
} from '@/types/research';

const API_BASE = '/api';

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(
      error.detail || `HTTP ${response.status}`,
      response.status,
      error
    );
  }
  return response.json();
}

/**
 * Start a new research workflow.
 */
export async function startResearch(
  request: ResearchRequest
): Promise<ResearchResponse> {
  const response = await fetch(`${API_BASE}/research/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  return handleResponse<ResearchResponse>(response);
}

/**
 * Get the current state of a research workflow.
 */
export async function getResearchState(
  threadId: string
): Promise<ResearchState> {
  const response = await fetch(`${API_BASE}/research/${threadId}/state`);
  return handleResponse<ResearchState>(response);
}

/**
 * Submit feedback on research results.
 */
export async function submitFeedback(
  threadId: string,
  feedback: string
): Promise<{ status: string; message: string }> {
  const response = await fetch(`${API_BASE}/research/${threadId}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ feedback }),
  });
  return handleResponse(response);
}

/**
 * Get the formatted research report.
 */
export async function getReport(threadId: string): Promise<{
  thread_id: string;
  account_name: string;
  report: string;
  opportunities_count: number;
  risks_count: number;
}> {
  const response = await fetch(`${API_BASE}/research/${threadId}/report`);
  return handleResponse(response);
}

/**
 * List all research threads.
 */
export async function listThreads(): Promise<{
  threads: ThreadSummary[];
  total: number;
}> {
  const response = await fetch(`${API_BASE}/research/list`);
  return handleResponse(response);
}

/**
 * Stop (pause) a running research workflow.
 * Preserves state so it can be resumed later from Previous Sessions.
 */
export async function stopResearch(threadId: string): Promise<{
  status: string;
  thread_id: string;
  message: string;
}> {
  const response = await fetch(`${API_BASE}/research/${threadId}/stop`, {
    method: 'POST',
  });
  return handleResponse(response);
}

/**
 * Permanently discard a research workflow.
 * Unlike stopResearch, this removes all state and cannot be resumed.
 */
export async function discardResearch(threadId: string): Promise<{
  status: string;
  thread_id: string;
  message: string;
}> {
  const response = await fetch(`${API_BASE}/research/${threadId}/discard`, {
    method: 'POST',
  });
  return handleResponse(response);
}

/**
 * Get per-node execution traces for the observability panel.
 * Returns timing and state summaries for each agent node.
 */
export async function getNodeTraces(
  threadId: string
): Promise<Record<string, NodeTrace>> {
  const response = await fetch(`${API_BASE}/research/${threadId}/node-traces`);
  return handleResponse<Record<string, NodeTrace>>(response);
}

/**
 * Health check.
 */
export async function healthCheck(): Promise<{
  status: string;
  version: string;
}> {
  const response = await fetch(`${API_BASE}/health`);
  return handleResponse(response);
}

export { ApiError };
