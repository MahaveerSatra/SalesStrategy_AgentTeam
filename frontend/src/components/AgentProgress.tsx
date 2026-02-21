/**
 * Live agent progress indicator component.
 */

import { CheckCircle2, Circle, Loader2, AlertCircle, Clock, Search, Lightbulb, Target, Shield } from 'lucide-react';
import type { Progress, NodeStatus } from '@/types/research';

interface AgentProgressProps {
  progress: Progress;
  nodeStatuses: Record<string, NodeStatus>;
  activeNode: string | null;
  nodeActivities?: Record<string, string>;  // Live activity messages
}

interface AgentStep {
  id: string;
  label: string;
  description: string;
  progressKey: keyof Progress;
  icon: React.ComponentType<{ className?: string }>;
  activeMessages: string[];  // Default messages when no specific activity
}

const AGENT_STEPS: AgentStep[] = [
  {
    id: 'coordinator_entry',
    label: 'Coordinator',
    description: 'Validating inputs & preparing research',
    progressKey: 'coordinator_complete',
    icon: Target,
    activeMessages: [
      'Validating account information...',
      'Checking industry alignment...',
      'Preparing research parameters...',
    ],
  },
  {
    id: 'gatherer',
    label: 'Gatherer',
    description: 'Searching web, jobs, and news',
    progressKey: 'gatherer_complete',
    icon: Search,
    activeMessages: [
      'Searching company news...',
      'Analyzing job postings...',
      'Gathering market signals...',
      'Collecting competitive intelligence...',
    ],
  },
  {
    id: 'identifier',
    label: 'Identifier',
    description: 'Matching products & extracting opportunities',
    progressKey: 'identifier_complete',
    icon: Lightbulb,
    activeMessages: [
      'Analyzing gathered signals...',
      'Matching products to needs...',
      'Identifying sales opportunities...',
      'Building opportunity rationale...',
    ],
  },
  {
    id: 'validator',
    label: 'Validator',
    description: 'Scoring confidence & assessing risks',
    progressKey: 'validator_complete',
    icon: Shield,
    activeMessages: [
      'Evaluating evidence strength...',
      'Scoring confidence levels...',
      'Assessing competitive risks...',
      'Validating opportunity quality...',
    ],
  },
];

function getStepStatus(
  step: AgentStep,
  progress: Progress,
  nodeStatuses: Record<string, NodeStatus>,
  activeNode: string | null
): NodeStatus {
  // Check explicit node status first
  if (nodeStatuses[step.id]) {
    return nodeStatuses[step.id];
  }

  // Check if this is the active node
  if (activeNode === step.id) {
    return 'running';
  }

  // Check progress
  if (progress[step.progressKey]) {
    return 'complete';
  }

  return 'idle';
}

function StatusIcon({ status }: { status: NodeStatus }) {
  switch (status) {
    case 'running':
      return <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />;
    case 'complete':
      return <CheckCircle2 className="w-5 h-5 text-green-600" />;
    case 'error':
      return <AlertCircle className="w-5 h-5 text-red-600" />;
    case 'waiting':
      return <Clock className="w-5 h-5 text-amber-600" />;
    default:
      return <Circle className="w-5 h-5 text-slate-300" />;
  }
}

// Animated message cycler for active agents
function useAnimatedMessage(messages: string[], isActive: boolean, interval = 2000) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setIndex(0);
      return;
    }

    const timer = setInterval(() => {
      setIndex(prev => (prev + 1) % messages.length);
    }, interval);

    return () => clearInterval(timer);
  }, [isActive, messages.length, interval]);

  return messages[index];
}

import { useState, useEffect } from 'react';

export function AgentProgress({
  progress,
  nodeStatuses,
  activeNode,
  nodeActivities = {},
}: AgentProgressProps) {
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-slate-600 uppercase tracking-wide">
        Agent Progress
      </h3>

      <div className="space-y-2">
        {AGENT_STEPS.map((step, index) => {
          const status = getStepStatus(step, progress, nodeStatuses, activeNode);
          const isActive = status === 'running';
          const Icon = step.icon;

          // Get activity message - either from SSE or use default animated message
          const sseActivity = nodeActivities[step.id];

          return (
            <div
              key={step.id}
              className={`
                rounded-lg transition-all overflow-hidden
                ${isActive ? 'bg-blue-50 border border-blue-200 shadow-sm' : 'bg-slate-50 border border-slate-100'}
              `}
            >
              {/* Main row */}
              <div className="flex items-center gap-3 p-3">
                <div className={`p-2 rounded-lg ${isActive ? 'bg-blue-100' : 'bg-slate-100'}`}>
                  <Icon className={`w-4 h-4 ${isActive ? 'text-blue-600' : 'text-slate-500'}`} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span
                      className={`font-medium ${
                        isActive ? 'text-blue-700' : status === 'complete' ? 'text-green-700' : 'text-slate-700'
                      }`}
                    >
                      {step.label}
                    </span>
                    {isActive && (
                      <span className="text-xs px-2 py-0.5 bg-blue-100 text-blue-700 rounded-full font-medium animate-pulse">
                        Running
                      </span>
                    )}
                    {status === 'complete' && (
                      <span className="text-xs px-2 py-0.5 bg-green-100 text-green-700 rounded-full font-medium">
                        Complete
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-slate-500 truncate">{step.description}</p>
                </div>

                <StatusIcon status={status} />
              </div>

              {/* Activity detail row - only show when active */}
              {isActive && (
                <div className="px-3 pb-3 pt-0">
                  <div className="ml-11 p-2 bg-blue-100/50 rounded border border-blue-200/50">
                    <ActivityDisplay
                      sseActivity={sseActivity}
                      defaultMessages={step.activeMessages}
                    />
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Human feedback indicator */}
      {activeNode === 'human_feedback' && (
        <div className="rounded-lg bg-amber-50 border border-amber-200 overflow-hidden">
          <div className="flex items-center gap-3 p-3">
            <div className="p-2 rounded-lg bg-amber-100">
              <Clock className="w-4 h-4 text-amber-600" />
            </div>
            <div className="flex-1">
              <span className="font-medium text-amber-800">Awaiting Your Feedback</span>
              <p className="text-xs text-amber-600">Review the report and provide guidance</p>
            </div>
            <Clock className="w-5 h-5 text-amber-600 animate-pulse" />
          </div>
        </div>
      )}
    </div>
  );
}

// Component to display activity with animation
function ActivityDisplay({
  sseActivity,
  defaultMessages
}: {
  sseActivity?: string;
  defaultMessages: string[];
}) {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    if (sseActivity) return; // Don't animate if we have real activity

    const timer = setInterval(() => {
      setMessageIndex(prev => (prev + 1) % defaultMessages.length);
    }, 2500);

    return () => clearInterval(timer);
  }, [sseActivity, defaultMessages.length]);

  const displayMessage = sseActivity || defaultMessages[messageIndex];

  return (
    <div className="flex items-center gap-2">
      <Loader2 className="w-3 h-3 text-blue-500 animate-spin flex-shrink-0" />
      <span className="text-xs text-blue-700 font-medium truncate">
        {displayMessage}
      </span>
    </div>
  );
}
