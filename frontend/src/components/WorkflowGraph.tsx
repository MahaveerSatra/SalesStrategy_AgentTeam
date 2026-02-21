/**
 * React Flow workflow visualization component.
 */

import { useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { AgentNode } from './AgentNode';
import type { NodeStatus, ResearchState } from '@/types/research';

interface WorkflowGraphProps {
  state: ResearchState | null;
  activeNode: string | null;
  nodeStatuses: Record<string, NodeStatus>;
}

const nodeTypes: NodeTypes = {
  agent: AgentNode,
};

// Base node definitions
const BASE_NODES: Node[] = [
  {
    id: 'coordinator_entry',
    type: 'agent',
    position: { x: 250, y: 0 },
    data: {
      label: 'Coordinator',
      description: 'Validate & prepare',
      icon: 'target',
      status: 'idle',
    },
  },
  {
    id: 'gatherer',
    type: 'agent',
    position: { x: 250, y: 120 },
    data: {
      label: 'Gatherer',
      description: 'Search web & jobs',
      icon: 'search',
      status: 'idle',
    },
  },
  {
    id: 'identifier',
    type: 'agent',
    position: { x: 250, y: 240 },
    data: {
      label: 'Identifier',
      description: 'Match products',
      icon: 'lightbulb',
      status: 'idle',
    },
  },
  {
    id: 'validator',
    type: 'agent',
    position: { x: 250, y: 360 },
    data: {
      label: 'Validator',
      description: 'Score & assess',
      icon: 'check',
      status: 'idle',
    },
  },
  {
    id: 'coordinator_exit',
    type: 'agent',
    position: { x: 250, y: 480 },
    data: {
      label: 'Report',
      description: 'Format results',
      icon: 'file',
      status: 'idle',
    },
  },
  {
    id: 'human_feedback',
    type: 'agent',
    position: { x: 480, y: 480 },
    data: {
      label: 'Your Review',
      description: 'Provide feedback',
      icon: 'user',
      status: 'idle',
    },
  },
];

const BASE_EDGES: Edge[] = [
  {
    id: 'e1',
    source: 'coordinator_entry',
    target: 'gatherer',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  {
    id: 'e2',
    source: 'gatherer',
    target: 'identifier',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  {
    id: 'e3',
    source: 'identifier',
    target: 'validator',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  {
    id: 'e4',
    source: 'validator',
    target: 'coordinator_exit',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  {
    id: 'e5',
    source: 'coordinator_exit',
    target: 'human_feedback',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  // Feedback loop edges (dashed)
  {
    id: 'e6',
    source: 'human_feedback',
    target: 'gatherer',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 1, strokeDasharray: '5,5' },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
  {
    id: 'e7',
    source: 'human_feedback',
    target: 'identifier',
    animated: false,
    style: { stroke: '#94a3b8', strokeWidth: 1, strokeDasharray: '5,5' },
    markerEnd: { type: MarkerType.ArrowClosed, color: '#94a3b8' },
  },
];

export function WorkflowGraph({
  state,
  activeNode,
  nodeStatuses,
}: WorkflowGraphProps) {
  // Update nodes with current status
  const nodes = useMemo(() => {
    return BASE_NODES.map(node => {
      let status: NodeStatus = nodeStatuses[node.id] || 'idle';

      // Override with active node
      if (activeNode === node.id) {
        status = 'running';
      }

      // Check progress from state
      if (state?.progress) {
        const progressKey = `${node.id.replace('coordinator_entry', 'coordinator').replace('coordinator_exit', 'coordinator')}_complete` as keyof typeof state.progress;
        if (state.progress[progressKey] && status === 'idle') {
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
          ...edge.markerEnd,
          color: isActive ? '#2563eb' : '#94a3b8',
        },
      };
    });
  }, [activeNode]);

  return (
    <div className="w-full h-full bg-slate-50 rounded-lg overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.5}
        maxZoom={1.5}
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={false}
        zoomOnScroll={false}
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
