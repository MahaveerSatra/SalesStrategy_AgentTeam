/**
 * TypeScript interfaces matching the backend ResearchState and API models.
 */

export type ResearchDepth = 'quick' | 'standard' | 'deep';

export type ResearchStatus =
  | 'pending'
  | 'running'
  | 'waiting_for_human'
  | 'completed'
  | 'stopped'  // Research paused by user, can be resumed
  | 'error';

export type WorkflowEventType =
  | 'workflow_started'
  | 'node_started'
  | 'node_completed'
  | 'state_update'
  | 'signal_found'
  | 'opportunity_found'
  | 'waiting_human'
  | 'feedback_received'
  | 'complete'
  | 'error'
  | 'heartbeat';

export interface Signal {
  source: string;
  signal_type: string;
  content: string;
  confidence: number;
  timestamp?: string;
  metadata: Record<string, unknown>;
}

export interface Opportunity {
  product_name: string;
  rationale: string;
  target_persona?: string;
  talking_points: string[];
  estimated_value?: string;
  risks: string[];
  confidence: string;
  confidence_score: number;
  evidence_count: number;
}

export interface Progress {
  coordinator_complete: boolean;
  gatherer_complete: boolean;
  identifier_complete: boolean;
  validator_complete: boolean;
  completed_agents: string[];
}

export interface ResearchRequest {
  account_name: string;
  industry: string;
  seller_name: string;
  region?: string;
  user_context?: string;
  research_depth: ResearchDepth;
}

export interface ResearchResponse {
  thread_id: string;
  status: ResearchStatus;
  message: string;
}

export interface ResearchState {
  thread_id: string;
  status: ResearchStatus;

  // Input params
  account_name: string;
  industry: string;
  seller_name: string;
  region?: string;
  user_context?: string;
  research_depth: string;

  // Progress
  progress: Progress;

  // Results
  signals: Signal[];
  opportunities: Opportunity[];
  validated_opportunities: Opportunity[];
  competitive_risks: string[];
  tech_stack: string[];

  // Human interaction
  waiting_for_human: boolean;
  human_question?: string;
  current_report?: string;

  // Metadata
  workflow_iteration: number;
  started_at?: string;
  last_updated?: string;
  error_messages: string[];

  // Observability
  langsmith_url?: string;
}

export interface WorkflowEvent {
  event: WorkflowEventType;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface ThreadSummary {
  thread_id: string;
  account_name: string;
  industry: string;
  status: ResearchStatus;
  started_at?: string;
  progress: Progress;
}

// Node trace types for observability panel
export interface NodeTraceSummary {
  // coordinator_entry
  account_name?: string;
  industry?: string;
  company_domain?: string;
  user_context?: string;
  // gatherer
  signals_count?: number;
  job_postings_count?: number;
  news_items_count?: number;
  signal_previews?: Array<{ source: string; signal_type: string; confidence: number }>;
  // identifier
  opportunities_count?: number;
  opportunity_previews?: Array<{ product_name: string; confidence_score: number }>;
  // validator
  validated_count?: number;
  risks_count?: number;
  tech_stack?: string[];
  risk_previews?: string[];
  // coordinator_exit
  report_length?: number;
  report_preview?: string;
  workflow_iteration?: number;
}

export interface NodeTrace {
  node_id: string;
  node_label: string;
  status: 'running' | 'complete' | 'error';
  start_time: string;
  end_time: string | null;
  duration_ms: number | null;
  summary: NodeTraceSummary;
}

// Node types for React Flow
export type NodeStatus = 'idle' | 'running' | 'complete' | 'error' | 'waiting';

export interface WorkflowNode {
  id: string;
  label: string;
  description: string;
  icon: string;
  status: NodeStatus;
  metrics?: {
    signals?: number;
    opportunities?: number;
    risks?: number;
  };
}
