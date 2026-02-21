/**
 * Human feedback modal for reviewing and responding to research results.
 */

import { useState } from 'react';
import { MessageSquare, Send, CheckCircle, RefreshCw, Search, X } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface HumanFeedbackProps {
  report: string;
  question?: string;
  onSubmit: (feedback: string) => void;
  onClose?: () => void;
  isLoading?: boolean;
}

const QUICK_RESPONSES = [
  {
    label: 'Approve',
    value: 'approved',
    icon: CheckCircle,
    color: 'bg-green-600 hover:bg-green-700',
  },
  {
    label: 'Dig Deeper',
    value: 'dig deeper on key initiatives',
    icon: Search,
    color: 'bg-blue-600 hover:bg-blue-700',
  },
  {
    label: 'Find Other Products',
    value: 'find other products that might fit',
    icon: RefreshCw,
    color: 'bg-amber-600 hover:bg-amber-700',
  },
];

export function HumanFeedback({
  report,
  question,
  onSubmit,
  onClose,
  isLoading = false,
}: HumanFeedbackProps) {
  const [customFeedback, setCustomFeedback] = useState('');

  const handleSubmit = (feedback: string) => {
    if (feedback.trim()) {
      onSubmit(feedback.trim());
      setCustomFeedback('');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
      <div className="w-full max-w-4xl max-h-[90vh] bg-white rounded-xl shadow-2xl border border-slate-200 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50 rounded-t-xl">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <MessageSquare className="w-5 h-5 text-blue-600" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Research Complete</h2>
              <p className="text-sm text-slate-500">Review the findings and provide feedback</p>
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Report Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-white">
          {question && (
            <div className="mb-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-blue-800">{question}</p>
            </div>
          )}

          <div className="markdown-content prose prose-slate max-w-none">
            <ReactMarkdown>{report}</ReactMarkdown>
          </div>
        </div>

        {/* Feedback Actions */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 space-y-4 rounded-b-xl">
          {/* Quick Response Buttons */}
          <div className="flex flex-wrap gap-2">
            {QUICK_RESPONSES.map(({ label, value, icon: Icon, color }) => (
              <button
                key={value}
                onClick={() => handleSubmit(value)}
                disabled={isLoading}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-white font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-sm ${color}`}
              >
                <Icon className="w-4 h-4" />
                {label}
              </button>
            ))}
          </div>

          {/* Custom Feedback Input */}
          <div className="flex gap-2">
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
              className="flex-1 px-4 py-3 bg-white border border-slate-300 rounded-lg text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
            <button
              onClick={() => handleSubmit(customFeedback)}
              disabled={isLoading || !customFeedback.trim()}
              className="px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white rounded-lg transition-all flex items-center gap-2 shadow-sm"
            >
              {isLoading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>

          <p className="text-xs text-slate-500 text-center">
            Tip: Say "approved" to finalize, or provide specific guidance to refine the research
          </p>
        </div>
      </div>
    </div>
  );
}
