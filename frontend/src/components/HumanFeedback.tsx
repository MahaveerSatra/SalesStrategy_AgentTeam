/**
 * Human feedback modal for reviewing and responding to research results.
 */

import { useState, useMemo } from 'react';
import { MessageSquare, Send, X, Download } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

// Helper to format coordinator questions with options
function formatQuestion(question: string): { mainQuestion: string; options: string[] } {
  // Check if question contains options (typically formatted with " - " prefix)
  const hasOptions = question.includes(' - ');

  if (!hasOptions) {
    return { mainQuestion: question, options: [] };
  }

  // Split by " - " to separate question from options
  const parts = question.split(' - ');
  const mainQuestion = parts[0].trim();
  const options = parts.slice(1).map(opt => opt.trim()).filter(opt => opt.length > 0);

  return { mainQuestion, options };
}

interface HumanFeedbackProps {
  report: string;
  question?: string;
  onSubmit: (feedback: string) => void;
  onClose?: () => void;
  isLoading?: boolean;
}

// Component to display formatted question with options
function FormattedQuestion({ question }: { question: string }) {
  const { mainQuestion, options } = useMemo(() => formatQuestion(question), [question]);

  return (
    <div className="mb-6 p-5 bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200 rounded-xl">
      <p className="text-blue-900 font-medium text-base leading-relaxed">
        {mainQuestion}
      </p>
      {options.length > 0 && (
        <>
          <ul className="mt-4 space-y-2">
            {options.map((option, idx) => (
              <li
                key={idx}
                className="flex items-center gap-3 text-blue-800 text-sm"
              >
                <span className="w-6 h-6 flex items-center justify-center bg-blue-100 text-blue-600 rounded-full text-xs font-semibold">
                  {idx + 1}
                </span>
                <span>{option}</span>
              </li>
            ))}
          </ul>
          <p className="mt-4 text-xs text-blue-600">
            Type a number (1-{options.length}) or provide your own answer below
          </p>
        </>
      )}
    </div>
  );
}

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

  const handleExport = () => {
    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `research_report_${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-sm">
      <div className="w-full max-w-4xl max-h-[90vh] bg-white rounded-xl shadow-2xl border border-slate-200 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-slate-50 rounded-t-xl">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${report ? 'bg-blue-100' : 'bg-amber-100'}`}>
              <MessageSquare className={`w-5 h-5 ${report ? 'text-blue-600' : 'text-amber-600'}`} />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                {report ? 'Research Complete' : 'Clarification Needed'}
              </h2>
              <p className="text-sm text-slate-500">
                {report ? 'Review the findings and provide feedback' : 'Please answer the question to continue'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {report && (
              <button
                onClick={handleExport}
                className="flex items-center gap-2 px-3 py-1.5 text-sm text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors"
                title="Download report as Markdown"
              >
                <Download className="w-4 h-4" />
                Export
              </button>
            )}
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Report Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-white">
          {question && (
            <FormattedQuestion question={question} />
          )}

          {report ? (
            <div className="markdown-content prose prose-slate max-w-none">
              <ReactMarkdown>{report}</ReactMarkdown>
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500">
              <p className="text-lg mb-2">The system needs your input to continue research.</p>
              <p className="text-sm">Please respond to the question above or provide additional context.</p>
            </div>
          )}
        </div>

        {/* Feedback Actions */}
        <div className="p-4 border-t border-slate-200 bg-slate-50 space-y-4 rounded-b-xl">
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
