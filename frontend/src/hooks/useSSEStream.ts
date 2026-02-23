/**
 * Hook for connecting to SSE stream and receiving workflow events.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { WorkflowEvent, WorkflowEventType, NodeStatus } from '@/types/research';

// Constants for heartbeat monitoring and auto-reconnect
const HEARTBEAT_CHECK_INTERVAL = 10000; // Check every 10 seconds
const HEARTBEAT_TIMEOUT = 30000; // Reconnect if no events for 30 seconds
const AUTO_RECONNECT_DELAY = 2000; // Wait 2 seconds before reconnecting

interface UseSSEStreamOptions {
  threadId: string | null;
  onEvent?: (event: WorkflowEvent) => void;
  onError?: (error: Error) => void;
  autoConnect?: boolean;
}

interface UseSSEStreamResult {
  events: WorkflowEvent[];
  activeNode: string | null;
  nodeStatuses: Record<string, NodeStatus>;
  nodeActivities: Record<string, string>;  // Live activity messages per node
  isConnected: boolean;
  error: Error | null;
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  resetStatuses: () => void;
}

// Map node names from backend to React Flow node IDs
const NODE_ID_MAP: Record<string, string> = {
  'COORDINATOR': 'coordinator_entry',
  'GATHERING': 'gatherer',
  'IDENTIFYING': 'identifier',
  'VALIDATING': 'validator',
  'REPORTING': 'coordinator_exit',
  'coordinator_entry': 'coordinator_entry',
  'gatherer': 'gatherer',
  'identifier': 'identifier',
  'validator': 'validator',
  'coordinator_exit': 'coordinator_exit',
};

export function useSSEStream({
  threadId,
  onEvent,
  onError,
  autoConnect = true,
}: UseSSEStreamOptions): UseSSEStreamResult {
  const [events, setEvents] = useState<WorkflowEvent[]>([]);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, NodeStatus>>({});
  const [nodeActivities, setNodeActivities] = useState<Record<string, string>>({});
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const lastEventTimeRef = useRef<number>(Date.now());
  const heartbeatIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const updateNodeStatus = useCallback((nodeId: string, status: NodeStatus) => {
    const mappedId = NODE_ID_MAP[nodeId] || nodeId;
    setNodeStatuses(prev => ({
      ...prev,
      [mappedId]: status,
    }));
  }, []);

  const updateNodeActivity = useCallback((nodeId: string, activity: string) => {
    const mappedId = NODE_ID_MAP[nodeId] || nodeId;
    setNodeActivities(prev => ({
      ...prev,
      [mappedId]: activity,
    }));
  }, []);

  const handleEvent = useCallback((event: WorkflowEvent) => {
    setEvents(prev => [...prev, event]);
    onEvent?.(event);

    // Update node statuses based on event type
    const eventType = event.event as WorkflowEventType;
    const nodeId = (event.data.node as string) || '';
    const mappedNodeId = NODE_ID_MAP[nodeId] || nodeId;

    switch (eventType) {
      case 'node_started':
        setActiveNode(mappedNodeId);
        updateNodeStatus(mappedNodeId, 'running');
        // Set initial activity description
        if (event.data.description) {
          updateNodeActivity(mappedNodeId, event.data.description as string);
        }
        break;
      case 'node_completed':
        // DON'T set activeNode to null - keep showing last node until next starts
        // This prevents status from flickering to "Initializing..."
        updateNodeStatus(mappedNodeId, 'complete');
        // Clear activity on completion
        updateNodeActivity(mappedNodeId, '');
        break;
      case 'state_update':
        // Update activity with details from state update
        if (event.data.activity && mappedNodeId) {
          updateNodeActivity(mappedNodeId, event.data.activity as string);
        }
        break;
      case 'signal_found':
        // Show signal being found
        if (event.data.signal_type) {
          const activity = `Found: ${event.data.signal_type}`;
          updateNodeActivity('gatherer', activity);
        }
        break;
      case 'opportunity_found':
        // Show opportunity being identified
        if (event.data.product_name) {
          const activity = `Identified: ${event.data.product_name}`;
          updateNodeActivity('identifier', activity);
        }
        break;
      case 'waiting_human':
        setActiveNode('human_feedback');
        updateNodeStatus('human_feedback', 'waiting');
        break;
      case 'complete':
        setActiveNode(null);
        break;
      case 'error':
        setActiveNode(null);
        if (mappedNodeId) {
          updateNodeStatus(mappedNodeId, 'error');
        }
        break;
    }
  }, [onEvent, updateNodeStatus, updateNodeActivity]);

  const connect = useCallback(() => {
    if (!threadId || eventSourceRef.current) return;

    const url = `/api/research/${threadId}/stream`;
    const eventSource = new EventSource(url);
    eventSourceRef.current = eventSource;

    eventSource.onopen = () => {
      setIsConnected(true);
      setError(null);
      lastEventTimeRef.current = Date.now();

      // Start heartbeat monitoring - check if events are still coming
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
      }
      heartbeatIntervalRef.current = setInterval(() => {
        const timeSinceLastEvent = Date.now() - lastEventTimeRef.current;
        if (timeSinceLastEvent > HEARTBEAT_TIMEOUT && eventSourceRef.current) {
          console.log('SSE heartbeat timeout, reconnecting...');
          // Close current connection and trigger reconnect
          eventSource.close();
          eventSourceRef.current = null;
          setIsConnected(false);
          // Auto-reconnect after delay
          reconnectTimeoutRef.current = setTimeout(() => {
            if (threadId) {
              connect();
            }
          }, AUTO_RECONNECT_DELAY);
        }
      }, HEARTBEAT_CHECK_INTERVAL);
    };

    eventSource.onmessage = (messageEvent) => {
      // Track last event time for heartbeat monitoring
      lastEventTimeRef.current = Date.now();
      try {
        const parsed = JSON.parse(messageEvent.data) as WorkflowEvent;
        handleEvent(parsed);
      } catch (err) {
        console.error('Failed to parse SSE event:', err);
      }
    };

    eventSource.onerror = () => {
      setIsConnected(false);
      const err = new Error('SSE connection error');
      setError(err);
      onError?.(err);
      eventSource.close();
      eventSourceRef.current = null;

      // Clear heartbeat interval
      if (heartbeatIntervalRef.current) {
        clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }

      // Auto-reconnect after delay
      reconnectTimeoutRef.current = setTimeout(() => {
        if (threadId) {
          console.log('SSE auto-reconnecting after error...');
          connect();
        }
      }, AUTO_RECONNECT_DELAY);
    };
  }, [threadId, handleEvent, onError]);

  const disconnect = useCallback(() => {
    // Clear heartbeat interval
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
    // Clear any pending reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    // Close the event source
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setIsConnected(false);
    }
  }, []);

  // Reset all statuses (call when feedback is submitted)
  const resetStatuses = useCallback(() => {
    setNodeStatuses({});
    setActiveNode(null);
    setNodeActivities({});
  }, []);

  // Reconnect SSE stream (call after feedback submission)
  const reconnect = useCallback(() => {
    disconnect();
    // Reset state for fresh connection
    setEvents([]);
    setNodeStatuses({});
    setNodeActivities({});
    setActiveNode(null);
    setError(null);
    // Small delay then reconnect
    setTimeout(() => {
      if (threadId) {
        connect();
      }
    }, 500);
  }, [disconnect, connect, threadId]);

  // Auto-connect when threadId changes
  useEffect(() => {
    if (autoConnect && threadId) {
      // Reset state for new thread
      setEvents([]);
      setActiveNode(null);
      setNodeStatuses({});
      setNodeActivities({});
      setError(null);

      connect();
    }

    return () => {
      disconnect();
    };
  }, [threadId, autoConnect, connect, disconnect]);

  return {
    events,
    activeNode,
    nodeStatuses,
    nodeActivities,
    isConnected,
    error,
    connect,
    disconnect,
    reconnect,
    resetStatuses,
  };
}
