/**
 * Main application component for Sales Research and Strategy Agent Team.
 */

import { useState } from 'react';
import { Users, RotateCcw, Github, Clock, PlayCircle, ChevronRight } from 'lucide-react';

import {
  ResearchForm,
  AgentProgress,
  ResultsPanel,
  HumanFeedback,
  WorkflowGraph,
} from '@/components';
import { useResearchWorkflow } from '@/hooks/useResearchWorkflow';
import { useSSEStream } from '@/hooks/useSSEStream';
import type { ResearchRequest, Progress, ThreadSummary } from '@/types/research';

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
      return 'text-blue-600 bg-blue-50';
    case 'waiting_for_human':
      return 'text-amber-600 bg-amber-50';
    case 'completed':
      return 'text-green-600 bg-green-50';
    case 'error':
      return 'text-red-600 bg-red-50';
    default:
      return 'text-slate-600 bg-slate-50';
  }
}

function SessionCard({ session, onResume }: { session: ThreadSummary; onResume: () => void }) {
  return (
    <button
      onClick={onResume}
      className="w-full text-left p-4 bg-white border border-slate-200 rounded-lg card-hover group"
    >
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <h3 className="font-medium text-slate-900 group-hover:text-blue-700">
            {session.account_name}
          </h3>
          <p className="text-sm text-slate-500">{session.industry}</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs px-2 py-1 rounded-full font-medium ${getStatusColor(session.status)}`}>
            {session.status === 'waiting_for_human' ? 'Needs Review' : session.status}
          </span>
          <ChevronRight className="w-5 h-5 text-slate-400 group-hover:text-blue-600 transition-colors" />
        </div>
      </div>
      {session.started_at && (
        <div className="flex items-center gap-1 mt-2 text-xs text-slate-400">
          <Clock className="w-3 h-3" />
          {formatDate(session.started_at)}
        </div>
      )}
    </button>
  );
}

function App() {
  const [showFeedback, setShowFeedback] = useState(false);

  const {
    threadId,
    state,
    isLoading,
    error,
    start,
    sendFeedback,
    reset,
    resumeThread,
    previousSessions,
    isLoadingSessions,
    isRunning,
    isWaitingForFeedback,
  } = useResearchWorkflow();

  const { activeNode, nodeStatuses, nodeActivities } = useSSEStream({
    threadId,
    autoConnect: true,
  });

  const handleStartResearch = async (request: ResearchRequest) => {
    await start(request);
  };

  const handleFeedback = async (feedback: string) => {
    await sendFeedback(feedback);
    setShowFeedback(false);
  };

  const handleReset = () => {
    reset();
    setShowFeedback(false);
  };

  const handleResume = (session: ThreadSummary) => {
    resumeThread(session.thread_id);
  };

  // Show feedback modal when waiting
  const shouldShowFeedback = isWaitingForFeedback && state?.current_report;

  // Default progress when no state
  const progress: Progress = state?.progress || {
    coordinator_complete: false,
    gatherer_complete: false,
    identifier_complete: false,
    validator_complete: false,
    completed_agents: [],
  };

  // Filter active sessions (running or waiting for human)
  const activeSessions = previousSessions.filter(
    s => s.status === 'running' || s.status === 'waiting_for_human'
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-blue-50">
      {/* Header */}
      <header className="header-gradient text-white sticky top-0 z-40 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm">
              <Users className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">
                Sales Research and Strategy Agent Team
              </h1>
              <p className="text-sm text-blue-100">
                Agentic Sales Strategy Team
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {threadId && (
              <button
                onClick={handleReset}
                className="flex items-center gap-2 px-4 py-2 text-sm text-white/90 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                New Research
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
        {!threadId ? (
          /* Initial Form View */
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-slate-900 mb-3">
                Research a Target Account
              </h2>
              <p className="text-slate-600 text-lg">
                Enter the account details below and let our AI agent team identify sales opportunities
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Research Form */}
              <div className="lg:col-span-2">
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <PlayCircle className="w-5 h-5 text-blue-600" />
                    Start New Research
                  </h3>
                  <ResearchForm onSubmit={handleStartResearch} isLoading={isLoading} />
                </div>
              </div>

              {/* Previous Sessions */}
              <div className="lg:col-span-1">
                <div className="bg-white rounded-xl p-6 border border-slate-200 shadow-sm">
                  <h3 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                    <Clock className="w-5 h-5 text-blue-600" />
                    Active Sessions
                  </h3>

                  {isLoadingSessions ? (
                    <div className="text-center py-4 text-slate-500">Loading...</div>
                  ) : activeSessions.length === 0 ? (
                    <div className="text-center py-4 text-slate-500">
                      No active sessions
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {activeSessions.map((session) => (
                        <SessionCard
                          key={session.thread_id}
                          session={session}
                          onResume={() => handleResume(session)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Research In Progress View - New Layout */
          <div className="space-y-6">
            {/* Top Section: Graph and Status Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Workflow Graph */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 border-b border-slate-100 bg-slate-50">
                  <h2 className="font-semibold text-slate-900">Agent Workflow</h2>
                  {state && (
                    <p className="text-sm text-slate-500 mt-1">
                      Researching: <span className="font-medium text-slate-700">{state.account_name}</span>
                    </p>
                  )}
                </div>
                <div className="h-[400px]">
                  <WorkflowGraph
                    state={state}
                    activeNode={activeNode}
                    nodeStatuses={nodeStatuses}
                  />
                </div>
              </div>

              {/* Right: Agent Status */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 border-b border-slate-100 bg-slate-50 flex items-center justify-between">
                  <h2 className="font-semibold text-slate-900">Agent Status</h2>
                  {isRunning && (
                    <span className="flex items-center gap-2 text-sm text-blue-600">
                      <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                      Processing...
                    </span>
                  )}
                  {isWaitingForFeedback && (
                    <button
                      onClick={() => setShowFeedback(true)}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
                    >
                      Review & Respond
                    </button>
                  )}
                </div>
                <div className="p-4">
                  <AgentProgress
                    progress={progress}
                    nodeStatuses={nodeStatuses}
                    activeNode={activeNode}
                    nodeActivities={nodeActivities}
                  />
                </div>
              </div>
            </div>

            {/* Bottom Section: Results */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="p-4 border-b border-slate-100 bg-slate-50">
                <h2 className="font-semibold text-slate-900">Research Results</h2>
              </div>
              <div className="p-6">
                <ResultsPanel
                  signals={state?.signals || []}
                  opportunities={state?.validated_opportunities || state?.opportunities || []}
                  risks={state?.competitive_risks || []}
                  techStack={state?.tech_stack || []}
                />
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                <p className="text-red-700 text-sm">{error.message}</p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Human Feedback Modal */}
      {shouldShowFeedback && showFeedback && (
        <HumanFeedback
          report={state?.current_report || ''}
          question={state?.human_question}
          onSubmit={handleFeedback}
          onClose={() => setShowFeedback(false)}
          isLoading={isLoading}
        />
      )}

      {/* Auto-show feedback when ready */}
      {shouldShowFeedback && !showFeedback && (
        <div className="fixed bottom-4 right-4 z-50">
          <button
            onClick={() => setShowFeedback(true)}
            className="flex items-center gap-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg shadow-lg shadow-blue-900/20 animate-pulse"
          >
            Research Complete - Review Now
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
