/**
 * Main application component for Sales Research and Strategy Agent Team.
 */

import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Users, Clock, ChevronDown, ChevronRight, Github, Heart, StopCircle, Play, Pause, Network, ArrowLeft } from 'lucide-react';

import {
  ResearchForm,
  WorkflowGraph,
  ReportView,
  HumanFeedback,
  NodeTracePanel,
} from '@/components';
import { useResearchWorkflow } from '@/hooks/useResearchWorkflow';
import { useSSEStream } from '@/hooks/useSSEStream';
import { getNodeTraces } from '@/lib/api';
import type { ResearchRequest, ThreadSummary } from '@/types/research';

// View state type for managing which view is shown
type ViewState = 'form' | 'research' | 'report' | 'graph';

function formatDate(dateStr?: string): string {
  if (!dateStr) return '';
  try {
    return new Date(dateStr).toLocaleString();
  } catch {
    return dateStr;
  }
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'running':
      return 'text-teal-600 bg-teal-50';
    case 'waiting_for_human':
      return 'text-amber-600 bg-amber-50';
    case 'completed':
      return 'text-emerald-600 bg-emerald-50';
    case 'stopped':
      return 'text-orange-600 bg-orange-50';
    case 'error':
      return 'text-rose-600 bg-rose-50';
    default:
      return 'text-zinc-600 bg-zinc-50';
  }
}

// Helper to get current step label from active node
function getCurrentStepLabel(activeNode: string | null, nodeActivities: Record<string, string>): string {
  if (!activeNode) return 'Initializing...';

  const labels: Record<string, string> = {
    'coordinator_entry': 'Coordinating research...',
    'gatherer': nodeActivities['gatherer'] || 'Gathering signals...',
    'identifier': nodeActivities['identifier'] || 'Identifying opportunities...',
    'validator': nodeActivities['validator'] || 'Validating findings...',
    'coordinator_exit': 'Generating report...',
    'human_feedback': 'Awaiting your feedback...',
  };

  return labels[activeNode] || 'Processing...';
}

function SessionCard({ session, onResume }: { session: ThreadSummary; onResume: () => void }) {
  const isStopped = session.status === 'stopped';

  return (
    <button
      onClick={onResume}
      className="w-full text-left p-4 card-hover group"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="font-medium text-zinc-900 group-hover:text-teal-600 transition-colors">
            {session.account_name}
          </h3>
          <p className="text-sm text-zinc-500">{session.industry}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${getStatusColor(session.status)}`}>
            {isStopped ? 'Paused' : session.status === 'waiting_for_human' ? 'Needs Review' : session.status}
          </span>
          {isStopped ? (
            <Play className="w-5 h-5 text-teal-500" />
          ) : (
            <ChevronRight className="w-5 h-5 text-zinc-400 group-hover:text-teal-500 transition-colors" />
          )}
        </div>
      </div>
      {session.started_at && (
        <div className="flex items-center gap-1 mt-2 text-xs text-zinc-400">
          <Clock className="w-3 h-3" />
          {formatDate(session.started_at)}
        </div>
      )}
    </button>
  );
}

// Collapsible Previous Sessions component
function PreviousSessions({
  sessions,
  onResume,
  isLoading
}: {
  sessions: ThreadSummary[];
  onResume: (session: ThreadSummary) => void;
  isLoading: boolean;
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (sessions.length === 0 && !isLoading) {
    return null;
  }

  return (
    <div className="mt-8">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="btn-collapse"
      >
        <ChevronDown
          className={`w-4 h-4 transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
        />
        Previous Sessions {!isLoading && `(${sessions.length})`}
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-2">
          {isLoading ? (
            <p className="text-sm text-zinc-500 py-2">Loading sessions...</p>
          ) : (
            sessions.map(session => (
              <SessionCard
                key={session.thread_id}
                session={session}
                onResume={() => onResume(session)}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

function App() {
  const [viewState, setViewState] = useState<ViewState>('form');
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [selectedTraceNode, setSelectedTraceNode] = useState<string | null>(null);
  const [fitViewTrigger, setFitViewTrigger] = useState(0);

  const {
    threadId,
    state,
    isLoading,
    error,
    start,
    sendFeedback,
    stop,
    discard,
    reset,
    resumeThread,
    previousSessions,
    isLoadingSessions,
    isRunning,
    isWaitingForFeedback,
  } = useResearchWorkflow();

  const { activeNode, nodeStatuses, nodeActivities, reconnect, resetStatuses } = useSSEStream({
    threadId,
    autoConnect: true,
  });

  // Fetch node traces when trace panel is open (polls during active research)
  const { data: nodeTraces = {} } = useQuery({
    queryKey: ['node-traces', threadId],
    queryFn: () => getNodeTraces(threadId!),
    enabled: !!threadId && selectedTraceNode !== null,
    staleTime: 5_000,
    refetchInterval: selectedTraceNode !== null && isRunning ? 3_000 : false,
  });

  // Auto-transition to report view when research completes with a full report
  useEffect(() => {
    if (isWaitingForFeedback && state?.current_report) {
      setViewState('report');
      setShowFeedbackModal(false);  // Don't show modal if we have report view
    }
  }, [isWaitingForFeedback, state?.current_report]);

  // Show feedback modal when waiting for human input (questions or report)
  useEffect(() => {
    if (isWaitingForFeedback && state?.status !== 'stopped' && !state?.current_report && state?.human_question) {
      // Show modal for clarifying questions (no report yet)
      setShowFeedbackModal(true);
    } else if (!isWaitingForFeedback) {
      setShowFeedbackModal(false);
    }
  }, [isWaitingForFeedback, state?.current_report, state?.human_question]);

  // Transition to research view when starting new research
  useEffect(() => {
    if (threadId && !isWaitingForFeedback) {
      setViewState('research');
    }
  }, [threadId, isWaitingForFeedback]);

  // Reset view state when no thread (but not when in graph preview mode)
  useEffect(() => {
    if (!threadId && viewState !== 'graph') {
      setViewState('form');
    }
  }, [threadId, viewState]);

  const handleStartResearch = async (request: ResearchRequest) => {
    setViewState('research');
    await start(request);
  };

  const handleFeedback = async (feedback: string) => {
    // Close the modal first
    setShowFeedbackModal(false);

    // Reset progress before submitting feedback
    resetStatuses();

    // Return to research view to show graph again
    setViewState('research');

    await sendFeedback(feedback);

    // Reconnect SSE to catch new events
    reconnect();
  };

  const handleReset = () => {
    reset();
    setViewState('form');
  };

  const handlePause = async () => {
    await stop();
    // Don't reset - keep showing the research view with paused state
    // User can click Resume to continue or Stop to discard
  };

  const handleDiscard = async () => {
    await discard();
    setViewState('form'); // Go back to form since research is gone
  };

  const handleResume = (session: ThreadSummary) => {
    resumeThread(session.thread_id);
    // If stopped, go to research view (will show Resume button)
    if (session.status === 'stopped') {
      setViewState('research');
    }
    // Other statuses will auto-transition based on state via useEffect
  };

  // If we're in graph preview mode, show the workflow graph without an active thread
  if (viewState === 'graph') {
    return (
      <div className="page-bg min-h-screen">
        <header className="header-gradient text-white sticky top-0 z-40 shadow-lg">
          <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
            <button
              onClick={handleReset}
              className="flex items-center gap-3 hover:opacity-90 transition-opacity"
            >
              <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
                <Users className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-xl font-bold tracking-tight">Agentic Sales Strategy Team</h1>
            </button>
            <button
              onClick={() => setViewState('form')}
              className="flex items-center gap-2 px-3 py-2 text-sm text-white/80 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
          </div>
        </header>
        <main className="max-w-7xl mx-auto px-4 py-4">
          <p className="text-center text-slate-500 text-sm mb-4">
            Agent workflow — start research to see it run live
          </p>
          <div className="h-[calc(100vh-140px)]">
            <WorkflowGraph state={null} activeNode={null} nodeStatuses={{}} />
          </div>
        </main>
      </div>
    );
  }

  // If we're in report view, show the full-page report
  if (viewState === 'report' && state?.current_report) {
    return (
      <ReportView
        report={state.current_report}
        signals={state.signals}
        question={state.human_question}
        accountName={state.account_name}
        onFeedback={handleFeedback}
        onHome={handleReset}
        onAgentGraph={() => setViewState('research')}
        isLoading={isLoading}
      />
    );
  }

  return (
    <div className="page-bg min-h-screen">
      {/* Header - Logo clickable for home navigation */}
      <header className="header-gradient text-white sticky top-0 z-40 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          {/* Logo - Clickable for home navigation */}
          <button
            onClick={handleReset}
            className="flex items-center gap-3 hover:opacity-90 transition-opacity"
          >
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
              <Users className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight">
              Agentic Sales Strategy Team
            </h1>
          </button>

          <div className="flex items-center gap-3">
            {viewState === 'form' && (
              <button
                onClick={() => setViewState('graph')}
                className="flex items-center gap-2 px-3 py-2 text-sm text-white/80 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                title="Preview the agent workflow"
              >
                <Network className="w-4 h-4" />
                View Workflow
              </button>
            )}
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
      <main className="max-w-7xl mx-auto px-4 py-8">
        {viewState === 'form' && !threadId ? (
          /* Initial Form View - Dopamine Design */
          <div className="max-w-3xl mx-auto relative overflow-visible">
            {/* Decorative Blobs */}
            <div className="blob-pink blob-animate" style={{ top: '-50px', left: '-100px' }} />
            <div className="blob-blue blob-animate" style={{ top: '200px', right: '-80px', animationDelay: '2s' }} />
            <div className="blob-teal blob-animate" style={{ bottom: '100px', left: '-60px', animationDelay: '4s' }} />

            {/* Hero Section */}
            <div className="hero-section text-center mb-10 relative z-10">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-gradient-to-r from-teal-50 to-pink-50 rounded-full border border-teal-100 mb-6">
                <span className="w-2 h-2 bg-teal-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-zinc-600">AI-Powered Sales Intelligence</span>
              </div>
              <h2 className="text-4xl font-bold mb-4">
                <span className="bg-gradient-to-r from-zinc-900 via-zinc-700 to-zinc-900 bg-clip-text text-transparent">
                  Research a Target Customer
                </span>
              </h2>
              <p className="text-zinc-500 text-lg max-w-lg mx-auto">
                Enter details below and let our AI agents identify
                <span className="text-teal-600 font-semibold"> sales opportunities</span>
              </p>
            </div>

            {/* Research Form - Playful Card with Stacked Effect */}
            <div className="form-wrapper relative z-10">
              <div className="form-stack">
                <div className="form-playful">
                  <ResearchForm onSubmit={handleStartResearch} isLoading={isLoading} />
                </div>
              </div>
            </div>

            {/* Previous Sessions - Collapsible Bottom Section */}
            <div className="relative z-10">
              <PreviousSessions
                sessions={previousSessions}
                onResume={handleResume}
                isLoading={isLoadingSessions}
              />
            </div>

            {/* Footer Credit */}
            <div className="footer-credit mt-12 relative z-10">
              <span>Made with </span>
              <Heart className="heart w-4 h-4 inline-block fill-current" />
              <span> by </span>
              <span className="font-semibold bg-gradient-to-r from-teal-600 to-pink-500 bg-clip-text text-transparent">
                Mahaveer
              </span>
            </div>
          </div>
        ) : (
          /* Research In Progress View - Full Width Graph with Status Line */
          <div className="space-y-4">
            {/* Status Bar - Title with Stop button on same row */}
            <div className="form-card p-4">
              <div className="flex flex-col gap-2">
                {/* Title row with Stop button */}
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold text-zinc-900">
                    {state?.status === 'stopped' ? 'Paused: ' : 'Researching: '}
                    <span className="text-teal-600">{state?.account_name}</span>
                  </h2>
                  {/* Show Resume/Stop buttons if paused, otherwise show Pause/Stop buttons */}
                  {state?.status === 'stopped' ? (
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => sendFeedback('continue')}
                        disabled={isLoading}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm text-teal-600 hover:text-teal-700 hover:bg-teal-50 rounded-lg transition-colors border border-teal-200 disabled:opacity-50"
                      >
                        <Play className="w-4 h-4" />
                        Resume
                      </button>
                      <button
                        onClick={handleDiscard}
                        disabled={isLoading}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm text-rose-600 hover:text-rose-700 hover:bg-rose-50 rounded-lg transition-colors border border-rose-200 disabled:opacity-50"
                      >
                        <StopCircle className="w-4 h-4" />
                        Stop
                      </button>
                    </div>
                  ) : isRunning && (
                    <div className="flex items-center gap-2">
                      {/* PAUSE button - saves checkpoint, can resume */}
                      <button
                        onClick={handlePause}
                        disabled={isLoading}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm text-amber-600 hover:text-amber-700 hover:bg-amber-50 rounded-lg transition-colors border border-amber-200 disabled:opacity-50"
                      >
                        <Pause className="w-4 h-4" />
                        Pause
                      </button>
                      {/* STOP button - discards research permanently */}
                      <button
                        onClick={handleDiscard}
                        disabled={isLoading}
                        className="flex items-center gap-2 px-3 py-1.5 text-sm text-rose-600 hover:text-rose-700 hover:bg-rose-50 rounded-lg transition-colors border border-rose-200 disabled:opacity-50"
                      >
                        <StopCircle className="w-4 h-4" />
                        Stop
                      </button>
                    </div>
                  )}
                </div>

                {/* Progress status as sub-heading */}
                {state?.status === 'stopped' ? (
                  <div className="flex items-center gap-2 text-orange-600">
                    <Pause className="w-4 h-4" />
                    <span className="font-medium text-sm">
                      Research paused - Resume to continue or Stop to discard
                    </span>
                  </div>
                ) : isRunning && (
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                    <span className="text-amber-700 font-medium text-sm">
                      {getCurrentStepLabel(activeNode, nodeActivities)}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Workflow Graph + Trace Panel — split when node is selected */}
            <div className="form-card overflow-hidden">
              <div className="flex gap-3 items-stretch overflow-hidden">
                <div className="flex-1 min-w-0 h-[320px]">
                  <WorkflowGraph
                    state={state}
                    activeNode={activeNode}
                    nodeStatuses={nodeStatuses}
                    fitViewTrigger={fitViewTrigger}
                    onReportClick={() => {
                      if (state?.current_report) {
                        setViewState('report');
                      }
                    }}
                    onAgentNodeClick={(nodeId) => {
                      setSelectedTraceNode(nodeId);
                      setFitViewTrigger(t => t + 1);
                    }}
                  />
                </div>
                {selectedTraceNode && (
                  <div className="w-[280px] flex-shrink-0 h-[320px]">
                    <NodeTracePanel
                      nodeId={selectedTraceNode}
                      traces={nodeTraces}
                      onClose={() => {
                        setSelectedTraceNode(null);
                        setFitViewTrigger(t => t + 1);
                      }}
                      langsmithUrl={state?.langsmith_url}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl">
                <p className="text-rose-700 text-sm">{error.message}</p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Human Feedback Modal - shown for clarifying questions */}
      {showFeedbackModal && state && (
        <HumanFeedback
          report={state.current_report || ''}
          question={state.human_question}
          onSubmit={handleFeedback}
          onClose={() => setShowFeedbackModal(false)}
          isLoading={isLoading}
        />
      )}
    </div>
  );
}

export default App;
