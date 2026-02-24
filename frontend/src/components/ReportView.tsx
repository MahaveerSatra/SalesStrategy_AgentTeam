/**
 * Full-page research report view component.
 * Replaces the entire app view when research is complete.
 */

import { useState } from 'react';
import { Download, Send, Github, Users, X, ExternalLink, ChevronDown, ChevronRight, ArrowLeft, CheckCircle, Search, RefreshCw, MessageSquare } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import type { Signal } from '@/types/research';

interface ReportViewProps {
  report: string;
  signals?: Signal[];
  question?: string;
  accountName?: string;
  onFeedback: (feedback: string) => void;
  onHome: () => void;
  onAgentGraph: () => void;
  isLoading?: boolean;
}

function SignalPanel({ signal, onClose }: { signal: Signal; onClose: () => void }) {
  const [showMetadata, setShowMetadata] = useState(false);

  // Extract URL from metadata if available
  const sourceUrl = signal.metadata?.url || signal.metadata?.source_url || null;

  return (
    <aside className="w-[350px] border-l border-zinc-200 bg-zinc-50 flex flex-col">
      <div className="p-4 border-b border-zinc-200 bg-white flex items-center justify-between">
        <h3 className="font-semibold text-zinc-900">Evidence Details</h3>
        <button
          onClick={onClose}
          className="p-1 text-zinc-400 hover:text-zinc-600 hover:bg-zinc-100 rounded transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-5">
        {/* 1. CONFIDENCE - First (most important) */}
        <div>
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">Confidence</span>
          <div className="mt-2 flex items-center gap-3">
            <div className="flex-1 h-2.5 bg-zinc-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-teal-500 rounded-full transition-all"
                style={{ width: `${signal.confidence * 100}%` }}
              />
            </div>
            <span className="text-lg font-semibold text-zinc-800">
              {Math.round(signal.confidence * 100)}%
            </span>
          </div>
        </div>

        {/* 2. CONTEXT/CONTENT - The actual information */}
        <div>
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">Context</span>
          <p className="mt-2 text-sm text-zinc-700 leading-relaxed bg-white p-3 rounded-lg border border-zinc-100">
            {signal.content}
          </p>
        </div>

        {/* 3. TYPE */}
        <div>
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">Type</span>
          <p className="mt-1 text-sm font-medium text-zinc-800 capitalize">
            {signal.signal_type.replace(/_/g, ' ')}
          </p>
        </div>

        {/* 4. SOURCE - Show actual URL if available */}
        <div>
          <span className="text-xs font-medium text-zinc-500 uppercase tracking-wide">Source</span>
          {sourceUrl ? (
            <a
              href={String(sourceUrl)}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 text-sm text-teal-600 hover:text-teal-700 hover:underline flex items-center gap-1 break-all"
            >
              {(() => {
                try {
                  return new URL(String(sourceUrl)).hostname;
                } catch {
                  return String(sourceUrl);
                }
              })()}
              <ExternalLink className="w-3 h-3 flex-shrink-0" />
            </a>
          ) : (
            <p className="mt-1 text-sm text-zinc-700">{signal.source}</p>
          )}
        </div>

        {/* 5. METADATA - Hidden by default with "Show more" */}
        {signal.metadata && Object.keys(signal.metadata).length > 0 && (
          <div>
            <button
              onClick={() => setShowMetadata(!showMetadata)}
              className="text-xs font-medium text-zinc-500 uppercase tracking-wide flex items-center gap-1 hover:text-teal-600 transition-colors"
            >
              {showMetadata ? (
                <ChevronDown className="w-3 h-3" />
              ) : (
                <ChevronRight className="w-3 h-3" />
              )}
              {showMetadata ? 'Show less' : 'Show more'}
            </button>

            {showMetadata && (
              <div className="mt-3 space-y-4 bg-white p-4 rounded-lg border border-zinc-100">
                {Object.entries(signal.metadata).map(([key, value]) => {
                  // Skip URL fields (already shown above)
                  if (key === 'url' || key === 'source_url') return null;

                  const label = key.replace(/_/g, ' ');

                  // Special handling for arrays (buying_signals, key_facts, keywords, etc.)
                  if (Array.isArray(value) && value.length > 0) {
                    return (
                      <div key={key}>
                        <h4 className="text-xs font-semibold text-teal-700 uppercase tracking-wide mb-2">
                          {label}
                        </h4>
                        <ul className="space-y-1.5">
                          {value.map((item, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-sm text-zinc-700">
                              <span className="text-teal-400 mt-0.5">•</span>
                              <span>{typeof item === 'string' ? item : JSON.stringify(item)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    );
                  }

                  // Special handling for objects
                  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                    return (
                      <div key={key}>
                        <h4 className="text-xs font-semibold text-teal-700 uppercase tracking-wide mb-2">
                          {label}
                        </h4>
                        <div className="bg-zinc-50 rounded-lg p-3 space-y-2">
                          {Object.entries(value as Record<string, unknown>).map(([subKey, subValue]) => (
                            <div key={subKey} className="flex flex-col gap-0.5">
                              <span className="text-xs text-zinc-500 capitalize">
                                {subKey.replace(/_/g, ' ')}
                              </span>
                              <span className="text-sm text-zinc-700">
                                {typeof subValue === 'string' ? subValue : JSON.stringify(subValue)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }

                  // Simple string/number values
                  if (value !== null && value !== undefined && value !== '') {
                    return (
                      <div key={key}>
                        <h4 className="text-xs font-semibold text-teal-700 uppercase tracking-wide mb-1">
                          {label}
                        </h4>
                        <p className="text-sm text-zinc-700">
                          {typeof value === 'string' ? value : JSON.stringify(value)}
                        </p>
                      </div>
                    );
                  }

                  return null;
                })}
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}

export function ReportView({
  report,
  signals = [],
  question,
  accountName,
  onFeedback,
  onHome,
  onAgentGraph,
  isLoading = false,
}: ReportViewProps) {
  const [customFeedback, setCustomFeedback] = useState('');
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);

  const handleSubmit = (feedback: string) => {
    if (feedback.trim()) {
      onFeedback(feedback.trim());
      setCustomFeedback('');
    }
  };

  const handleExport = () => {
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${accountName || 'research'}_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      {/* Header - Logo clickable for home navigation */}
      <header className="header-gradient text-white sticky top-0 z-40 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          {/* Logo - Clickable for home navigation */}
          <button
            onClick={onHome}
            className="flex items-center gap-3 hover:opacity-90 transition-opacity"
          >
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
              <Users className="w-6 h-6 text-white" />
            </div>
            <div className="text-left">
              <h1 className="text-xl font-bold tracking-tight">Research Report</h1>
              {accountName && (
                <p className="text-sm text-white/80">{accountName}</p>
              )}
            </div>
          </button>

          <div className="flex items-center gap-3">
            <button
              onClick={onAgentGraph}
              className="flex items-center gap-2 px-4 py-2 text-sm text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="Back to agent graph"
            >
              <ArrowLeft className="w-4 h-4" />
              Agent Graph
            </button>
            <button
              onClick={handleExport}
              className="flex items-center gap-2 px-4 py-2 text-sm text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="Download report as Markdown"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
            <a
              href="https://github.com/MahaveerSatra/SalesStrategy_AgentTeam"
              target="_blank"
              rel="noopener noreferrer"
              className="p-2 text-white/80 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="View on GitHub"
            >
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Report Content */}
        <main className={`flex-1 ${selectedSignal ? 'max-w-[calc(100%-350px)]' : ''}`}>
          <div className="max-w-4xl mx-auto px-8 py-8">
            {/* Action Guide — shown when coordinator is waiting for feedback */}
            {question && (
              <div className="mb-6 p-4 bg-teal-50 border border-teal-200 rounded-xl space-y-3">
                <p className="text-xs font-semibold text-teal-700 uppercase tracking-wide">
                  What would you like to do next?
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={() => handleSubmit('approved')}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-3 py-2 bg-teal-600 hover:bg-teal-700 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors text-left"
                  >
                    <CheckCircle className="w-4 h-4 flex-shrink-0" />
                    Approve report
                  </button>
                  <button
                    onClick={() => setCustomFeedback('dig deeper on ')}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-3 py-2 bg-white hover:bg-slate-50 disabled:opacity-50 text-slate-700 text-sm font-medium rounded-lg border border-slate-200 transition-colors text-left"
                  >
                    <Search className="w-4 h-4 flex-shrink-0 text-slate-500" />
                    Dig deeper
                  </button>
                  <button
                    onClick={() => handleSubmit('different products')}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-3 py-2 bg-white hover:bg-slate-50 disabled:opacity-50 text-slate-700 text-sm font-medium rounded-lg border border-slate-200 transition-colors text-left"
                  >
                    <RefreshCw className="w-4 h-4 flex-shrink-0 text-slate-500" />
                    Different products
                  </button>
                  <button
                    onClick={() => setCustomFeedback('')}
                    disabled={isLoading}
                    className="flex items-center gap-2 px-3 py-2 bg-white hover:bg-slate-50 disabled:opacity-50 text-slate-700 text-sm font-medium rounded-lg border border-slate-200 transition-colors text-left"
                  >
                    <MessageSquare className="w-4 h-4 flex-shrink-0 text-slate-500" />
                    Custom feedback ↓
                  </button>
                </div>
              </div>
            )}

            {/* Report Markdown */}
            <div className="prose-report">
              <ReactMarkdown
                components={{
                  // Make links open in new tab
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-teal-600 hover:text-teal-700 inline-flex items-center gap-1"
                    >
                      {children}
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  ),
                }}
              >
                {report}
              </ReactMarkdown>
            </div>

            {/* Evidence Section - Clean list with context and clickable links */}
            {signals.length > 0 && (
              <div className="mt-8 pt-6 border-t border-zinc-200">
                <h3 className="text-sm font-semibold text-zinc-600 uppercase tracking-wide mb-4">
                  Evidence ({signals.length})
                </h3>
                <ul className="space-y-3">
                  {signals.map((signal, index) => {
                    const sourceUrl = signal.metadata?.url || signal.metadata?.source_url || null;
                    return (
                      <li key={index} className="flex items-start gap-2 text-sm text-zinc-700">
                        <span className="text-teal-500 mt-1">•</span>
                        <span className="flex-1 leading-relaxed">
                          {signal.content}
                          {sourceUrl ? (
                            <button
                              onClick={() => setSelectedSignal(signal)}
                              className="ml-2 inline-flex items-center gap-1 text-teal-600 hover:text-teal-700 hover:underline"
                            >
                              <ExternalLink className="w-3 h-3" />
                              <span className="text-xs">View details</span>
                            </button>
                          ) : (
                            <button
                              onClick={() => setSelectedSignal(signal)}
                              className="ml-2 text-teal-600 hover:text-teal-700 hover:underline text-xs"
                            >
                              View details
                            </button>
                          )}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Feedback Section */}
            <div className="mt-8 pt-6 border-t border-zinc-200">
              <h3 className="text-lg font-semibold text-zinc-900 mb-4">
                Provide Feedback
              </h3>


              {/* Custom Feedback Input */}
              <div className="flex gap-3">
                <input
                  type="text"
                  value={customFeedback}
                  onChange={(e) => setCustomFeedback(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && customFeedback.trim()) {
                      handleSubmit(customFeedback);
                    }
                  }}
                  placeholder="Or type custom feedback... (e.g., 'focus on cloud initiatives')"
                  disabled={isLoading}
                  className="flex-1 px-4 py-3 bg-white border-2 border-zinc-200 rounded-xl text-zinc-900 placeholder-zinc-400 focus:outline-none focus:border-teal-500 focus:ring-2 focus:ring-teal-500/20 disabled:opacity-50 transition-all"
                />
                <button
                  onClick={() => handleSubmit(customFeedback)}
                  disabled={isLoading || !customFeedback.trim()}
                  className="px-5 py-3 bg-teal-600 hover:bg-teal-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white rounded-xl transition-all flex items-center gap-2 shadow-sm"
                >
                  {isLoading ? (
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>

              <p className="mt-3 text-sm text-zinc-500">
                Say "approved" to finalize the research, or provide specific guidance to refine the findings.
              </p>
            </div>
          </div>
        </main>

        {/* Signal Side Panel */}
        {selectedSignal && (
          <SignalPanel
            signal={selectedSignal}
            onClose={() => setSelectedSignal(null)}
          />
        )}
      </div>
    </div>
  );
}
