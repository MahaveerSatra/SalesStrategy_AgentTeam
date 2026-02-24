/**
 * Custom React Flow node component for workflow visualization.
 */

import { memo } from 'react';
import { Handle, Position, type NodeProps } from 'reactflow';
import {
  Target,
  Search,
  Lightbulb,
  CheckCircle,
  FileText,
  User,
  Loader2,
  AlertCircle,
  Clock,
  Shield
} from 'lucide-react';
import type { NodeStatus } from '@/types/research';

interface AgentNodeData {
  label: string;
  description: string;
  icon: string;
  status?: NodeStatus;
  handles?: {
    left?: boolean;
    right?: boolean;
    top?: boolean;
    bottom?: boolean;
  };
  metrics?: {
    signals?: number;
    opportunities?: number;
    risks?: number;
  };
}

const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  target: Target,
  search: Search,
  lightbulb: Lightbulb,
  check: CheckCircle,
  file: FileText,
  user: User,
  shield: Shield,
};

function StatusIndicator({ status }: { status: NodeStatus }) {
  switch (status) {
    case 'running':
      return (
        <div className="absolute -top-1 -right-1">
          <Loader2 className="w-4 h-4 text-amber-600 animate-spin" />
        </div>
      );
    case 'complete':
      return (
        <div className="absolute -top-1 -right-1">
          <CheckCircle className="w-4 h-4 text-green-600" />
        </div>
      );
    case 'error':
      return (
        <div className="absolute -top-1 -right-1">
          <AlertCircle className="w-4 h-4 text-red-600" />
        </div>
      );
    case 'waiting':
      return (
        <div className="absolute -top-1 -right-1">
          <Clock className="w-4 h-4 text-amber-600 animate-pulse" />
        </div>
      );
    default:
      return null;
  }
}

function AgentNodeComponent({ data, selected }: NodeProps<AgentNodeData>) {
  const status = data.status || 'idle';
  const IconComponent = ICONS[data.icon] || Target;

  const statusClasses = {
    idle: 'bg-white border-slate-200',
    running: 'bg-amber-50 border-amber-400 shadow-lg shadow-amber-500/20 animate-pulse',
    complete: 'bg-green-50 border-green-400',
    error: 'bg-red-50 border-red-400',
    waiting: 'bg-amber-50 border-amber-400',
  };

  return (
    <div
      className={`
        relative px-4 py-3 rounded-lg border-2 w-[180px] transition-all duration-300
        ${statusClasses[status]}
        ${selected ? 'ring-2 ring-blue-500 ring-offset-2 ring-offset-white' : ''}
      `}
    >
      {/* Left handle for incoming connections (horizontal layout) */}
      {data.handles?.left && (
        <Handle
          type="target"
          position={Position.Left}
          id="left"
          className="!w-3 !h-3 !bg-slate-300 !border-2 !border-slate-400"
        />
      )}

      {/* Top handle for vertical connections (incoming from above) */}
      {data.handles?.top && (
        <Handle
          type="target"
          position={Position.Top}
          id="top"
          className="!w-3 !h-3 !bg-slate-300 !border-2 !border-slate-400"
        />
      )}

      <StatusIndicator status={status} />

      <div className="flex items-center gap-3">
        <div
          className={`
            p-2 rounded-lg
            ${status === 'running' ? 'bg-amber-100' : 'bg-slate-100'}
          `}
        >
          <IconComponent
            className={`w-5 h-5 ${
              status === 'running' ? 'text-amber-600' : 'text-slate-600'
            }`}
          />
        </div>

        <div>
          <div
            className={`font-semibold text-sm ${
              status === 'running' ? 'text-amber-800' : 'text-slate-800'
            }`}
          >
            {data.label}
          </div>
          <div className="text-xs text-slate-500 max-w-[120px] truncate">
            {data.description}
          </div>
        </div>
      </div>

      {/* Metrics */}
      {data.metrics && Object.keys(data.metrics).length > 0 && (
        <div className="mt-2 pt-2 border-t border-slate-200 flex gap-3 text-xs">
          {data.metrics.signals !== undefined && (
            <span className="text-blue-600 font-medium">
              {data.metrics.signals} signals
            </span>
          )}
          {data.metrics.opportunities !== undefined && (
            <span className="text-amber-600 font-medium">
              {data.metrics.opportunities} opps
            </span>
          )}
          {data.metrics.risks !== undefined && (
            <span className="text-red-600 font-medium">
              {data.metrics.risks} risks
            </span>
          )}
        </div>
      )}

      {/* Bottom handle for vertical connections (outgoing to below) */}
      {data.handles?.bottom && (
        <Handle
          type="source"
          position={Position.Bottom}
          id="bottom"
          className="!w-3 !h-3 !bg-slate-300 !border-2 !border-slate-400"
        />
      )}

      {/* Right handle for outgoing connections (horizontal layout) */}
      {data.handles?.right && (
        <Handle
          type="source"
          position={Position.Right}
          id="right"
          className="!w-3 !h-3 !bg-slate-300 !border-2 !border-slate-400"
        />
      )}
    </div>
  );
}

export const AgentNode = memo(AgentNodeComponent);
