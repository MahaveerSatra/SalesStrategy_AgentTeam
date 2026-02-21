/**
 * Hook for managing research workflow state.
 * Persists thread_id to localStorage for session recovery.
 */

import { useState, useCallback, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  startResearch,
  getResearchState,
  submitFeedback,
  listThreads,
} from '@/lib/api';
import type { ResearchRequest, ResearchState, ThreadSummary } from '@/types/research';

const STORAGE_KEY = 'sales_research_thread_id';

function getStoredThreadId(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function setStoredThreadId(threadId: string | null): void {
  try {
    if (threadId) {
      localStorage.setItem(STORAGE_KEY, threadId);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // Ignore storage errors
  }
}

interface UseResearchWorkflowResult {
  // Current state
  threadId: string | null;
  state: ResearchState | null;
  isLoading: boolean;
  error: Error | null;

  // Actions
  start: (request: ResearchRequest) => Promise<string>;
  sendFeedback: (feedback: string) => Promise<void>;
  reset: () => void;
  resumeThread: (threadId: string) => void;

  // Previous sessions
  previousSessions: ThreadSummary[];
  isLoadingSessions: boolean;
  refreshSessions: () => void;

  // Status helpers
  isRunning: boolean;
  isWaitingForFeedback: boolean;
  isComplete: boolean;
}

export function useResearchWorkflow(): UseResearchWorkflowResult {
  // Initialize from localStorage
  const [threadId, setThreadId] = useState<string | null>(() => getStoredThreadId());
  const queryClient = useQueryClient();

  // Persist threadId to localStorage whenever it changes
  useEffect(() => {
    setStoredThreadId(threadId);
  }, [threadId]);

  // Query for current state
  const {
    data: state,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['research', threadId],
    queryFn: () => getResearchState(threadId!),
    enabled: !!threadId,
    refetchInterval: (query) => {
      const data = query.state.data;
      // Stop polling when waiting for human or completed
      if (data?.waiting_for_human || data?.status === 'completed') {
        return false;
      }
      // Poll every 2 seconds while running
      return data?.status === 'running' ? 2000 : false;
    },
    retry: 2,
  });

  // Query for previous sessions
  const {
    data: sessionsData,
    isLoading: isLoadingSessions,
    refetch: refreshSessions,
  } = useQuery({
    queryKey: ['research-sessions'],
    queryFn: listThreads,
    staleTime: 30000, // 30 seconds
  });

  // Mutation to start research
  const startMutation = useMutation({
    mutationFn: startResearch,
    onSuccess: (data) => {
      setThreadId(data.thread_id);
    },
  });

  // Mutation to submit feedback
  const feedbackMutation = useMutation({
    mutationFn: ({ threadId, feedback }: { threadId: string; feedback: string }) =>
      submitFeedback(threadId, feedback),
    onSuccess: () => {
      // Invalidate to refetch state
      queryClient.invalidateQueries({ queryKey: ['research', threadId] });
    },
  });

  const start = useCallback(
    async (request: ResearchRequest): Promise<string> => {
      const result = await startMutation.mutateAsync(request);
      return result.thread_id;
    },
    [startMutation]
  );

  const sendFeedback = useCallback(
    async (feedback: string): Promise<void> => {
      if (!threadId) throw new Error('No active research thread');
      await feedbackMutation.mutateAsync({ threadId, feedback });
    },
    [threadId, feedbackMutation]
  );

  const reset = useCallback(() => {
    setThreadId(null);
    queryClient.removeQueries({ queryKey: ['research'] });
  }, [queryClient]);

  const resumeThread = useCallback((id: string) => {
    setThreadId(id);
    queryClient.invalidateQueries({ queryKey: ['research', id] });
  }, [queryClient]);

  // Status helpers
  const isRunning = state?.status === 'running';
  const isWaitingForFeedback = state?.waiting_for_human ?? false;
  const isComplete = state?.status === 'completed';

  return {
    threadId,
    state: state ?? null,
    isLoading: isLoading || startMutation.isPending || feedbackMutation.isPending,
    error: error ?? startMutation.error ?? feedbackMutation.error ?? null,
    start,
    sendFeedback,
    reset,
    resumeThread,
    previousSessions: sessionsData?.threads ?? [],
    isLoadingSessions,
    refreshSessions,
    isRunning,
    isWaitingForFeedback,
    isComplete,
  };
}
