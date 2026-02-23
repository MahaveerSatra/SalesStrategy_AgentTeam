/**
 * React Flow workflow visualization component.
 */

import { useMemo, useCallback } from 'react';
import ReactFlow, {
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  MarkerType,
  type NodeMouseHandler,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { AgentNode } from './AgentNode';
import type { NodeStatus, ResearchState } from '@/types/research';

interface WorkflowGraphProps {
  state: ResearchState | null;
  activeNode: string | null;
  nodeStatuses: Record<string, NodeStatus>;
  onReportClick?: () => void;
}

const nodeTypes: NodeTypes = {
  agent: AgentNode,
};

// Base node definitions - HORIZONTAL PARALLEL layout with fan-out/fan-in
// Flow: User ↔ Coordinator (Entry) → [Gatherer, Identifier, Validator] → Coordinator (Exit) → Report
// Parallel agents stacked vertically, with sequential vertical connections between them
// Positions spread out horizontally for cleaner arrow routing
const BASE_NODES: Node[] = [
  {
    id: 'user',
    type: 'agent',
    position: { x: 0, y: 100 },
    data: {
      label: 'You',
      description: 'Start & feedback',
      icon: 'user',
      status: 'idle',
      handles: { right: true },
    },
  },
  {
    id: 'coordinator_entry',
    type: 'agent',
    position: { x: 220, y: 100 },
    data: {
      label: 'Coordinator',
      description: 'Plan & clarify',
      icon: 'target',
      status: 'idle',
      handles: { left: true, right: true },
    },
  },
  {
    id: 'gatherer',
    type: 'agent',
    position: { x: 480, y: 0 },
    data: {
      label: 'Gatherer',
      description: 'Search signals',
      icon: 'search',
      status: 'idle',
      handles: { left: true, right: true, bottom: true },
    },
  },
  {
    id: 'identifier',
    type: 'agent',
    position: { x: 480, y: 100 },
    data: {
      label: 'Identifier',
      description: 'Find products',
      icon: 'lightbulb',
      status: 'idle',
      handles: { left: true, right: true, top: true, bottom: true },
    },
  },
  {
    id: 'validator',
    type: 'agent',
    position: { x: 480, y: 200 },
    data: {
      label: 'Validator',
      description: 'Score & rank',
      icon: 'shield',
      status: 'idle',
      handles: { left: true, right: true, top: true },
    },
  },
  {
    id: 'coordinator_exit',
    type: 'agent',
    position: { x: 740, y: 100 },
    data: {
      label: 'Coordinator',
      description: 'Review & report',
      icon: 'target',
      status: 'idle',
      handles: { left: true, right: true },
    },
  },
  {
    id: 'report',
    type: 'agent',
    position: { x: 960, y: 100 },
    data: {
      label: 'Report',
      description: 'Final output',
      icon: 'file',
      status: 'idle',
      clickable: true,
      handles: { left: true },
    },
  },
];

// HORIZONTAL PARALLEL edges with fan-out, sequential vertical, and fan-in connections
// Flow: User ↔ Coordinator → [Gatherer → Identifier → Validator] → Coordinator (Exit) → Report
// Using 'smoothstep' type for clean right-angle connections (no curves)
// Horizontal edges use sourceHandle='right' and targetHandle='left' for proper left-to-right flow
const BASE_EDGES: Edge[] = [
  // User → Coordinator Entry (form submission) - horizontal
  {
    id: 'e-user-coord',
    source: 'user',
    target: 'coordinator_entry',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Coordinator Entry → Gatherer (fan-out) - horizontal right→left
  {
    id: 'e-coord-gather',
    source: 'coordinator_entry',
    target: 'gatherer',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Coordinator Entry → Identifier (fan-out) - horizontal right→left
  {
    id: 'e-coord-ident',
    source: 'coordinator_entry',
    target: 'identifier',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Coordinator Entry → Validator (fan-out) - horizontal right→left
  {
    id: 'e-coord-valid',
    source: 'coordinator_entry',
    target: 'validator',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Gatherer → Identifier (sequential vertical) - straight down
  {
    id: 'e-gather-ident',
    source: 'gatherer',
    target: 'identifier',
    type: 'smoothstep',
    sourceHandle: 'bottom',
    targetHandle: 'top',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Identifier → Validator (sequential vertical) - straight down
  {
    id: 'e-ident-valid',
    source: 'identifier',
    target: 'validator',
    type: 'smoothstep',
    sourceHandle: 'bottom',
    targetHandle: 'top',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Gatherer → Coordinator Exit (fan-in) - horizontal right→left
  {
    id: 'e-gather-coord-exit',
    source: 'gatherer',
    target: 'coordinator_exit',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Identifier → Coordinator Exit (fan-in) - horizontal right→left
  {
    id: 'e-ident-coord-exit',
    source: 'identifier',
    target: 'coordinator_exit',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Validator → Coordinator Exit (fan-in) - horizontal right→left
  {
    id: 'e-valid-coord-exit',
    source: 'validator',
    target: 'coordinator_exit',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Coordinator Exit → Report - horizontal
  {
    id: 'e-coord-exit-report',
    source: 'coordinator_exit',
    target: 'report',
    type: 'smoothstep',
    sourceHandle: 'right',
    targetHandle: 'left',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
];

export function WorkflowGraph({
  state,
  activeNode,
  nodeStatuses,
  onReportClick,
}: WorkflowGraphProps) {
  // Helper to get progress key for a node
  const getProgressKey = (nodeId: string): string | null => {
    switch (nodeId) {
      case 'coordinator_entry':
        return 'coordinator_complete';
      case 'gatherer':
        return 'gatherer_complete';
      case 'identifier':
        return 'identifier_complete';
      case 'validator':
        return 'validator_complete';
      case 'coordinator_exit':
        return 'validator_complete'; // Exit shows complete when validator done
      default:
        return null;
    }
  };

  // Update nodes with current status
  // Key behaviors for human-in-loop:
  // 1. User node: YELLOW (waiting) when clarifying questions, GREEN after answering
  // 2. Report node: Only GREEN when final report is ready (not during clarifying questions)
  // 3. Coordinator Exit: Shows running when generating report, complete when report ready
  // 4. Other nodes: Standard running/complete behavior
  const nodes = useMemo(() => {
    return BASE_NODES.map(node => {
      let status: NodeStatus = 'idle'; // Default to idle

      // USER NODE - Human-in-loop behavior
      if (node.id === 'user') {
        if (state?.waiting_for_human) {
          // User is waiting to provide feedback (yellow)
          status = 'waiting';
        } else if (state?.progress?.coordinator_complete) {
          // User has provided input and research is progressing
          status = 'complete';
        }
      }
      // REPORT NODE - Only green for FINAL report
      else if (node.id === 'report') {
        // Only show complete when there IS a report AND waiting for human
        if (state?.current_report && state?.waiting_for_human) {
          status = 'complete';
        }
      }
      // COORDINATOR EXIT - Shows when reviewing/generating report
      else if (node.id === 'coordinator_exit') {
        if (activeNode === 'coordinator_exit') {
          status = 'running';
        } else if (state?.current_report) {
          status = 'complete';
        }
      }
      // For other nodes: only highlight if THIS node is the active node
      else if (activeNode === node.id) {
        status = 'running';
      }
      // Check explicit node status from SSE
      else if (nodeStatuses[node.id]) {
        status = nodeStatuses[node.id];
      }
      // Mark as complete only if progress indicates completion AND no active node
      else if (state?.progress && !activeNode) {
        const progressKey = getProgressKey(node.id);
        if (progressKey && state.progress[progressKey as keyof typeof state.progress]) {
          status = 'complete';
        }
      }

      // Add metrics based on node type
      let metrics = {};
      if (state) {
        if (node.id === 'gatherer' && state.signals.length > 0) {
          metrics = { signals: state.signals.length };
        }
        if (node.id === 'identifier' && state.opportunities.length > 0) {
          metrics = { opportunities: state.opportunities.length };
        }
        if (node.id === 'validator') {
          if (state.validated_opportunities.length > 0 || state.competitive_risks.length > 0) {
            metrics = {
              opportunities: state.validated_opportunities.length,
              risks: state.competitive_risks.length,
            };
          }
        }
      }

      return {
        ...node,
        data: {
          ...node.data,
          status,
          metrics: Object.keys(metrics).length > 0 ? metrics : undefined,
        },
      };
    });
  }, [state, activeNode, nodeStatuses]);

  // Update edges - animate active transitions
  const edges = useMemo(() => {
    return BASE_EDGES.map(edge => {
      const isActive = activeNode === edge.target;
      return {
        ...edge,
        animated: isActive,
        style: {
          ...edge.style,
          stroke: isActive ? '#2563eb' : '#94a3b8',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isActive ? '#2563eb' : '#94a3b8',
        },
      };
    });
  }, [activeNode]);

  // Handle node clicks - only Report node is clickable when report is ready
  const handleNodeClick: NodeMouseHandler = useCallback(
    (_event, node) => {
      if (node.id === 'report' && state?.current_report && onReportClick) {
        onReportClick();
      }
    },
    [state?.current_report, onReportClick]
  );

  // Determine if report node should have clickable cursor
  const hasReport = Boolean(state?.current_report);

  return (
    <div className="w-full h-full bg-slate-50 rounded-lg overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.15 }}
        minZoom={0.4}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={hasReport}
        onNodeClick={handleNodeClick}
        panOnDrag={false}
        zoomOnScroll={false}
        className={hasReport ? 'cursor-pointer' : ''}
      >
        <Background color="#e2e8f0" gap={20} size={1} />
        <Controls
          showInteractive={false}
          className="!bg-white !border-slate-200 !shadow-md"
        />
      </ReactFlow>
    </div>
  );
}
