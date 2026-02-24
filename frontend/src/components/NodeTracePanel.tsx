/**
 * Node Trace Panel — observability drawer for agent execution details.
 *
 * Shows per-node timing, inputs, and outputs when user clicks an agent node
 * in the workflow graph. Displayed as a 35% right panel alongside the graph.
 */

import { useEffect } from 'react';
import {
  ChevronRight,
  Clock,
  CheckCircle,
  Loader2,
  AlertCircle,
  ArrowUpRight,
  Layers,
  Inbox,
  Send,
} from 'lucide-react';
import type { NodeTrace } from '@/types/research';

interface NodeTracePanelProps {
  nodeId: string;
  traces: Record<string, NodeTrace>;
  onClose: () => void;
  langsmithUrl?: string | null;
}

// Human-readable node labels
const NODE_LABELS: Record<string, string> = {
  coordinator_entry: 'Coordinator',
  gatherer: 'Gatherer',
  identifier: 'Identifier',
  validator: 'Validator',
  coordinator_exit: 'Report Coordinator',
};

function StatusBadge({ status }: { status: NodeTrace['status'] }) {
  if (status === 'running') {
    return (
      <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-700">
        <Loader2 className="w-3 h-3 animate-spin" />
        Running
      </span>
    );
  }
  if (status === 'complete') {
    return (
      <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-teal-100 text-teal-700">
        <CheckCircle className="w-3 h-3" />
        Complete
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700">
      <AlertCircle className="w-3 h-3" />
      Error
    </span>
  );
}

function formatDuration(ms: number | null): string {
  if (ms === null) return '—';
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function formatTime(iso: string | null): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return '—';
  }
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 75 ? 'bg-teal-500' : pct >= 50 ? 'bg-amber-400' : 'bg-red-400';
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-slate-500 w-8 text-right">{pct}%</span>
    </div>
  );
}

function CoordinatorSummary({ summary }: { summary: NodeTrace['summary'] }) {
  return (
    <div className="space-y-2 text-xs">
      {summary.account_name && (
        <div className="flex justify-between">
          <span className="text-slate-500">Account</span>
          <span className="font-medium text-slate-800">{summary.account_name}</span>
        </div>
      )}
      {summary.industry && (
        <div className="flex justify-between">
          <span className="text-slate-500">Industry</span>
          <span className="font-medium text-slate-800">{summary.industry}</span>
        </div>
      )}
      {summary.company_domain && (
        <div className="flex justify-between">
          <span className="text-slate-500">Domain</span>
          <span className="font-medium text-slate-800">{summary.company_domain}</span>
        </div>
      )}
      {summary.user_context && (
        <div className="pt-1">
          <p className="text-slate-500 mb-1">Context</p>
          <p className="text-slate-700 bg-slate-50 rounded p-2 leading-relaxed">
            {summary.user_context}
          </p>
        </div>
      )}
    </div>
  );
}

function GathererSummary({ summary }: { summary: NodeTrace['summary'] }) {
  return (
    <div className="space-y-3 text-xs">
      <div className="grid grid-cols-3 gap-2">
        <div className="bg-blue-50 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-blue-700">{summary.signals_count ?? 0}</div>
          <div className="text-blue-600">Signals</div>
        </div>
        <div className="bg-slate-50 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-slate-700">{summary.job_postings_count ?? 0}</div>
          <div className="text-slate-500">Job Posts</div>
        </div>
        <div className="bg-slate-50 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-slate-700">{summary.news_items_count ?? 0}</div>
          <div className="text-slate-500">News</div>
        </div>
      </div>
      {summary.signal_previews && summary.signal_previews.length > 0 && (
        <div>
          <p className="text-slate-500 mb-1.5">Top signals</p>
          <div className="space-y-2">
            {summary.signal_previews.map((s, i) => (
              <div key={i} className="bg-slate-50 rounded p-2 space-y-1">
                <p className="text-slate-700 text-[11px] leading-snug line-clamp-2">{s.source}</p>
                {s.signal_type && (
                  <span className="text-[10px] text-slate-400 truncate block">
                    {(() => { try { return new URL(s.signal_type).hostname; } catch { return s.signal_type; } })()}
                  </span>
                )}
                <ConfidenceBar value={s.confidence} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function IdentifierSummary({ summary }: { summary: NodeTrace['summary'] }) {
  return (
    <div className="space-y-3 text-xs">
      <div className="bg-amber-50 rounded-lg p-2 text-center">
        <div className="text-xl font-bold text-amber-700">{summary.opportunities_count ?? 0}</div>
        <div className="text-amber-600">Opportunities identified</div>
      </div>
      {summary.opportunity_previews && summary.opportunity_previews.length > 0 && (
        <div>
          <p className="text-slate-500 mb-1.5">Top opportunities</p>
          <div className="space-y-2">
            {summary.opportunity_previews.map((o, i) => (
              <div key={i} className="bg-slate-50 rounded p-2 space-y-1">
                <span className="font-medium text-slate-700">{o.product_name}</span>
                <ConfidenceBar value={o.confidence_score} />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ValidatorSummary({ summary }: { summary: NodeTrace['summary'] }) {
  const total = (summary.opportunities_count ?? 0) || (summary.validated_count ?? 0);
  const validated = summary.validated_count ?? 0;
  const pct = total > 0 ? Math.round((validated / total) * 100) : 0;

  return (
    <div className="space-y-3 text-xs">
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-teal-50 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-teal-700">{validated}</div>
          <div className="text-teal-600">Validated</div>
        </div>
        <div className="bg-red-50 rounded-lg p-2 text-center">
          <div className="text-lg font-bold text-red-700">{summary.risks_count ?? 0}</div>
          <div className="text-red-600">Risks</div>
        </div>
      </div>
      {total > 0 && (
        <div>
          <div className="flex justify-between mb-1">
            <span className="text-slate-500">Pass rate</span>
            <span className="font-medium text-slate-700">{pct}%</span>
          </div>
          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
            <div className="h-full bg-teal-500 rounded-full" style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
      {summary.tech_stack && summary.tech_stack.length > 0 && (
        <div>
          <p className="text-slate-500 mb-1.5">Tech stack detected</p>
          <div className="flex flex-wrap gap-1">
            {summary.tech_stack.map((tech, i) => (
              <span key={i} className="px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded text-xs">
                {tech}
              </span>
            ))}
          </div>
        </div>
      )}
      {summary.risk_previews && summary.risk_previews.length > 0 && (
        <div>
          <p className="text-slate-500 mb-1.5">Top risks</p>
          <ul className="space-y-1">
            {summary.risk_previews.map((r, i) => (
              <li key={i} className="text-red-600 bg-red-50 rounded p-1.5 leading-snug">{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ReportCoordinatorSummary({ summary }: { summary: NodeTrace['summary'] }) {
  return (
    <div className="space-y-2 text-xs">
      {summary.report_length !== undefined && (
        <div className="flex justify-between">
          <span className="text-slate-500">Report size</span>
          <span className="font-medium text-slate-800">
            {(summary.report_length / 1000).toFixed(1)}k chars
          </span>
        </div>
      )}
      {summary.workflow_iteration !== undefined && summary.workflow_iteration > 1 && (
        <div className="flex justify-between">
          <span className="text-slate-500">Iterations</span>
          <span className="font-medium text-slate-800">{summary.workflow_iteration}</span>
        </div>
      )}
      {summary.report_preview && (
        <div className="pt-1">
          <p className="text-slate-500 mb-1">Preview</p>
          <p className="text-slate-700 bg-slate-50 rounded p-2 leading-relaxed font-mono text-[10px] whitespace-pre-wrap line-clamp-6">
            {summary.report_preview}
          </p>
        </div>
      )}
    </div>
  );
}

function NodeSummaryContent({ nodeId, summary }: { nodeId: string; summary: NodeTrace['summary'] }) {
  if (Object.keys(summary).length === 0) {
    return (
      <p className="text-xs text-slate-400 italic">
        {nodeId === 'coordinator_entry' ? 'Gathering inputs...' : 'Processing...'}
      </p>
    );
  }

  switch (nodeId) {
    case 'coordinator_entry':
      return <CoordinatorSummary summary={summary} />;
    case 'gatherer':
      return <GathererSummary summary={summary} />;
    case 'identifier':
      return <IdentifierSummary summary={summary} />;
    case 'validator':
      return <ValidatorSummary summary={summary} />;
    case 'coordinator_exit':
      return <ReportCoordinatorSummary summary={summary} />;
    default:
      return null;
  }
}

export function NodeTracePanel({ nodeId, traces, onClose, langsmithUrl }: NodeTracePanelProps) {
  const trace = traces[nodeId];
  const label = NODE_LABELS[nodeId] || nodeId;

  // Close on ESC key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div className="h-full flex flex-col bg-white border border-slate-200 rounded-lg overflow-hidden shadow-sm">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-slate-200 bg-slate-50 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-slate-200 transition-colors shrink-0"
            title="Collapse panel (ESC)"
          >
            <ChevronRight className="w-4 h-4 text-slate-500" />
          </button>
          <span className="font-semibold text-sm text-slate-800 truncate">{label}</span>
        </div>
        {trace && <StatusBadge status={trace.status} />}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {!trace ? (
          <div className="p-4 text-center text-xs text-slate-400">
            <Layers className="w-6 h-6 mx-auto mb-2 text-slate-300" />
            No trace data yet
          </div>
        ) : (
          <div className="p-3 space-y-3">
            {/* Timing */}
            <div className="bg-slate-50 rounded-lg p-2.5 space-y-1.5 text-xs">
              <div className="flex items-center gap-1 text-slate-500 font-medium mb-1">
                <Clock className="w-3.5 h-3.5" />
                Timing
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1">
                <span className="text-slate-500">Started</span>
                <span className="font-medium text-slate-700">{formatTime(trace.start_time)}</span>
                <span className="text-slate-500">Finished</span>
                <span className="font-medium text-slate-700">{formatTime(trace.end_time)}</span>
                <span className="text-slate-500">Duration</span>
                <span className={`font-bold ${trace.status === 'complete' ? 'text-teal-700' : 'text-amber-600'}`}>
                  {formatDuration(trace.duration_ms)}
                </span>
              </div>
            </div>

            {/* Output summary */}
            <div>
              <div className="flex items-center gap-1 text-xs text-slate-500 font-medium mb-2">
                <Send className="w-3.5 h-3.5" />
                {trace.status === 'running' ? 'In progress' : 'Output'}
              </div>
              <NodeSummaryContent nodeId={nodeId} summary={trace.summary} />
            </div>
          </div>
        )}
      </div>

      {/* Footer — LangSmith link (Phase B) */}
      {langsmithUrl && (
        <div className="px-3 py-2 border-t border-slate-200 shrink-0">
          <a
            href={langsmithUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-1.5 w-full py-1.5 text-xs text-teal-600 hover:text-teal-700 hover:bg-teal-50 rounded-lg transition-colors font-medium"
          >
            <Inbox className="w-3.5 h-3.5" />
            View Full Trace in LangSmith
            <ArrowUpRight className="w-3 h-3" />
          </a>
        </div>
      )}
    </div>
  );
}
