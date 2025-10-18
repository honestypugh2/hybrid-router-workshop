// Enhanced types for the hybrid router application (mirroring streamlit_multiturn_demo.py)

export interface ConversationExchange {
  timestamp: Date;
  user_message: string;
  ai_response: string;
  source: string;
  response_time: number;
  exchange_number: number;
  metadata?: ConversationMetadata;
  model_switched?: boolean;
}

export interface ConversationMetadata {
  strategy?: string;
  context_used?: boolean;
  routing_info?: RoutingMetadata;
  api_connected?: boolean;
  error?: boolean;
  error_message?: string;
  context_length?: number;
  query_length?: number;
  router_type?: string;
  routing_reason?: string;
  complexity_score?: number;
  routing_system?: string;
}

export interface SessionStats {
  start_time: Date;
  total_exchanges: number;
  model_switches: number;
  last_model: string | null;
  context_preserved: number;
  fallback_used: number;
}

export interface RoutingMetadata {
  strategy: string;
  target: string;
  confidence: number;
  reason: string;
  source: string;
  response_time: number;
  success: boolean;
  timestamp: string;
  analysis?: AnalysisMetadata;
  context_used?: boolean;
}

export interface AnalysisMetadata {
  router_used?: string;
  is_greeting?: boolean;
  is_calculation?: boolean;
  ml_confidence?: number;
  complexity_level?: string;
  routing_decision?: string;
  bert_confidence?: number;
  phi_confidence?: number;
}

export interface ConversationInsights {
  total_exchanges: number;
  avg_response_time: number;
  fastest_response: number;
  slowest_response: number;
  source_distribution: Record<string, number>;
  model_switches: number;
  context_usage_rate: number;
  session_duration: number;
}

export interface SystemStatus {
  availableRouters: Record<string, boolean>;
  systemHealth: 'healthy' | 'degraded' | 'error';
  hybrid_router_targets?: Record<string, boolean>;
  capabilities?: SystemCapabilities;
}

export interface SystemCapabilities {
  available_targets: Record<string, boolean>;
  hybrid_routing: boolean;
  bert_routing: boolean;
  phi_routing: boolean;
  context_management: boolean;
  multi_turn_support: boolean;
}

export interface ConversationContext {
  session_id: string;
  conversation_history: ConversationExchange[];
  session_stats: SessionStats;
  max_exchanges: number;
}

export interface QueryRequest {
  query: string;
  strategy: RouterStrategy;
  session_id?: string;
  context?: boolean;
}

export interface QueryResponse {
  response: string;
  source: string;
  responseTime: number;
  metadata: RoutingMetadata;
}

export interface PerformanceMetrics {
  timestamp: Date;
  response_time: number;
  source: string;
  context_used: boolean;
  strategy: RouterStrategy;
  success: boolean;
}

export interface ExampleQuery {
  category: 'simple' | 'moderate' | 'complex';
  text: string;
  description: string;
  expected_source?: string;
}

// Enhanced router strategies matching streamlit implementation
export type RouterStrategy = 'hybrid' | 'rule_based' | 'bert' | 'phi';

// Enhanced model sources matching streamlit implementation  
export type ModelSource = 'local' | 'apim' | 'foundry' | 'azure' | 'cloud' | 'mock' | 'error';

// Router configuration options
export interface RouterConfig {
  strategy: RouterStrategy;
  confidence_threshold?: number;
  context_window?: number;
  enable_fallback?: boolean;
  enable_context?: boolean;
}

// Enhanced analytics data
export interface AnalyticsData {
  routing_stats: Record<ModelSource, number>;
  performance_history: PerformanceMetrics[];
  conversation_insights: ConversationInsights | null;
  system_status: SystemStatus;
  session_summary: SessionSummary;
}

export interface SessionSummary {
  duration_minutes: number;
  total_queries: number;
  successful_queries: number;
  error_rate: number;
  most_used_source: string;
  avg_confidence: number;
  context_effectiveness: number;
}