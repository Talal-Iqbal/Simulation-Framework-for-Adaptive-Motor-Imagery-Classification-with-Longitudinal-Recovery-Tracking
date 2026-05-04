export type RunState = "idle" | "running" | "paused" | "ended";

export interface TrialPredictResponse {
  trial_idx: number;
  y_true: string;
  y_pred: string;
  correct: boolean;
  confidence: number;
  margin: number;
  accepted: boolean;
  reject_reasons: string[];
  timestamp_s: number;
}

export interface BatchPredictResponse {
  n_trials: number;
  accepted: number;
  rejected: number;
  accuracy_all: number | null;
  accuracy_accepted: number | null;
  rejection_breakdown: Record<string, number>;
  results: TrialPredictResponse[];
}

export interface HealthResponse {
  status: string;
  version: string;
  models: Record<string, string | null>;
}

export interface SessionAnalyzeResponse {
  cluster: number;
  predicted_r: number;
  silhouette: number | null;
}

export interface SessionFeatureVector {
  erd_mu_c3_mean: number;
  erd_mu_c4_mean: number;
  erd_lat_mu_mean: number;
  erd_lat_mu_abs_mean: number;
  lda_conf_mean: number;
  lda_margin_mean: number;
  lda_margin_std: number;
  mu_ratio_mean: number;
  motor_relative_mean: number;
  frob_dist_to_baseline: number;
  intertrial_cov_var: number;
  lda_session_acc: number;
  p2p_max_mean: number;
}

export interface SessionSnapshot {
  sessionId: string;
  subjectId: number;
  acceptedRate: number;
  gatedAccuracy: number;
  rejectRate: number;
  frobDistToBaseline: number;
  intertrialCovVar: number;
  estimatedR: number;
  clusterId: number;
  assistanceState: "normal" | "watch" | "assist";
}

export type NavKey =
  | "overview"
  | "session"
  | "calibration"
  | "layer6"
  | "layer7"
  | "diagnostics"
  | "subjects";
