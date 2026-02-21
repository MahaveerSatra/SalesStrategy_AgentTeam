/**
 * Results panel showing signals, opportunities, and risks.
 */

import { Signal as SignalIcon, Lightbulb, AlertTriangle, TrendingUp } from 'lucide-react';
import type { Signal, Opportunity } from '@/types/research';

interface ResultsPanelProps {
  signals: Signal[];
  opportunities: Opportunity[];
  risks: string[];
  techStack: string[];
}

function ConfidenceBadge({ score }: { score: number }) {
  const percentage = Math.round(score * 100);
  let colorClass = 'bg-red-100 text-red-700';

  if (percentage >= 80) {
    colorClass = 'bg-green-100 text-green-700';
  } else if (percentage >= 60) {
    colorClass = 'bg-amber-100 text-amber-700';
  }

  return (
    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>
      {percentage}%
    </span>
  );
}

function SignalCard({ signal }: { signal: Signal }) {
  return (
    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
      <div className="flex items-start justify-between gap-2 mb-2">
        <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded font-medium">
          {signal.signal_type}
        </span>
        <ConfidenceBadge score={signal.confidence} />
      </div>
      <p className="text-sm text-slate-700 line-clamp-3">{signal.content}</p>
      <p className="text-xs text-slate-400 mt-2">Source: {signal.source}</p>
    </div>
  );
}

function OpportunityCard({ opportunity }: { opportunity: Opportunity }) {
  return (
    <div className="p-4 bg-white rounded-lg border border-slate-200 shadow-sm">
      <div className="flex items-start justify-between gap-2 mb-3">
        <h4 className="font-semibold text-slate-900">{opportunity.product_name}</h4>
        <ConfidenceBadge score={opportunity.confidence_score} />
      </div>

      <p className="text-sm text-slate-600 mb-3">{opportunity.rationale}</p>

      {opportunity.target_persona && (
        <p className="text-xs text-slate-500 mb-2">
          <span className="text-slate-400">Target:</span> {opportunity.target_persona}
        </p>
      )}

      {opportunity.talking_points.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-medium text-slate-500 mb-1">Talking Points:</p>
          <ul className="space-y-1">
            {opportunity.talking_points.slice(0, 3).map((point, i) => (
              <li key={i} className="text-xs text-slate-600 flex items-start gap-1">
                <span className="text-blue-600 mt-0.5">•</span>
                <span className="line-clamp-2">{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {opportunity.risks.length > 0 && (
        <div className="mt-3 pt-3 border-t border-slate-100">
          <p className="text-xs text-red-600">
            {opportunity.risks.length} risk{opportunity.risks.length > 1 ? 's' : ''} identified
          </p>
        </div>
      )}
    </div>
  );
}

export function ResultsPanel({
  signals,
  opportunities,
  risks,
  techStack,
}: ResultsPanelProps) {
  return (
    <div className="space-y-6">
      {/* Signals Section */}
      <section>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
          <SignalIcon className="w-4 h-4 text-blue-600" />
          Signals ({signals.length})
        </h3>
        {signals.length > 0 ? (
          <div className="grid gap-3 max-h-64 overflow-y-auto pr-2">
            {signals.map((signal, i) => (
              <SignalCard key={i} signal={signal} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400 italic">No signals collected yet...</p>
        )}
      </section>

      {/* Opportunities Section */}
      <section>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
          <Lightbulb className="w-4 h-4 text-amber-500" />
          Opportunities ({opportunities.length})
        </h3>
        {opportunities.length > 0 ? (
          <div className="space-y-3">
            {opportunities.map((opp, i) => (
              <OpportunityCard key={i} opportunity={opp} />
            ))}
          </div>
        ) : (
          <p className="text-sm text-slate-400 italic">No opportunities identified yet...</p>
        )}
      </section>

      {/* Risks Section */}
      <section>
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
          <AlertTriangle className="w-4 h-4 text-red-500" />
          Competitive Risks ({risks.length})
        </h3>
        {risks.length > 0 ? (
          <ul className="space-y-2">
            {risks.map((risk, i) => (
              <li
                key={i}
                className="text-sm text-slate-700 p-3 bg-red-50 border border-red-100 rounded-lg flex items-start gap-2"
              >
                <AlertTriangle className="w-4 h-4 text-red-500 mt-0.5 flex-shrink-0" />
                {risk}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-400 italic">No risks identified yet...</p>
        )}
      </section>

      {/* Tech Stack */}
      {techStack.length > 0 && (
        <section>
          <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-600 uppercase tracking-wide mb-3">
            <TrendingUp className="w-4 h-4 text-green-600" />
            Tech Stack
          </h3>
          <div className="flex flex-wrap gap-2">
            {techStack.map((tech, i) => (
              <span
                key={i}
                className="px-2 py-1 text-xs bg-slate-100 text-slate-700 rounded font-medium"
              >
                {tech}
              </span>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
