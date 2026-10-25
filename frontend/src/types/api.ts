export interface CensorSegment {
  text: string;
  censored: boolean;
}

export interface SeverityInfo {
  score: number;
  band: 'Low' | 'Medium' | 'High' | 'Critical';
}

export interface ModelPrediction {
  model_id: string;
  model_name: string;
  role: string;
  available: boolean;
  device: string;
  probabilities: Record<string, number>;
  per_label_flags: Record<string, boolean>;
  normalized_scores: Record<string, number>;
  severity: SeverityInfo;
  action: 'Remove' | 'Hide with warning' | 'Flag for review' | 'Allow' | string;
  action_color: 'red' | 'orange' | 'yellow' | 'green' | 'gray';
  needs_review: boolean;
  reasoning: string[];
  execution_time_ms: number;
  error_message?: string | null;
}

export interface HeadlineRecommendation {
  model_id: string;
  model_name: string;
  action: 'Remove' | 'Hide with warning' | 'Flag for review' | 'Allow' | string;
  action_color: 'red' | 'orange' | 'yellow' | 'green';
  severity: SeverityInfo;
  needs_review: boolean;
  reasoning: string[];
}

export interface PredictResponse {
  id: string;
  timestamp: string;
  text_length: number;
  original_text: string;
  censored_text: string;
  censored_segments: CensorSegment[];
  has_profanity: boolean;
  policy_version: string;
  headline: HeadlineRecommendation;
  models: Record<string, ModelPrediction>;
}

export interface CommentRecord {
  id: string;
  timestamp: string;
  text_length: number;
  original_text: string;
  censored_text: string;
  censored_segments: CensorSegment[];
  has_profanity: boolean;
  policy_version: string;
  headline: HeadlineRecommendation;
  models: Record<string, ModelPrediction>;
}

export interface CommentsListResponse {
  items: CommentRecord[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export interface ModelInfo {
  id: string;
  name: string;
  role: string;
  description: string;
  available: boolean;
  device: string;
  thresholds: Record<string, number>;
  remove_cuts: Record<string, number | null>;
  error_message?: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  mongodb_connected: boolean;
  mongodb_uri_masked: string;
  models_available: number;
  total_models: number;
  gpu_available: boolean;
  gpu_device_name?: string | null;
  uptime_seconds: number;
}
