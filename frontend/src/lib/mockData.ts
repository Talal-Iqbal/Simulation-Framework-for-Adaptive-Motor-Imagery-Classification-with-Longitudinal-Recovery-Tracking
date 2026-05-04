import type {
  SessionAnalyzeResponse,
  SessionFeatureVector,
  TrialPredictResponse
} from "../types/api";

const rejectReasons = ["accepted", "low_conf", "artifact", "flatline", "high_noise"];

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function mockTrial(trialIdx: number, prevTimestamp: number): TrialPredictResponse {
  const accepted = Math.random() > 0.2;
  const confidence = Number((0.35 + Math.random() * 0.6).toFixed(3));
  const margin = Number(((Math.random() - 0.5) * 2).toFixed(3));
  const yTrue = Math.random() > 0.5 ? "left_hand" : "right_hand";
  const yPred = confidence > 0.52 ? yTrue : yTrue === "left_hand" ? "right_hand" : "left_hand";

  return {
    trial_idx: trialIdx,
    y_true: yTrue,
    y_pred: yPred,
    correct: yTrue === yPred,
    confidence,
    margin,
    accepted,
    reject_reasons: accepted ? [] : [pick(rejectReasons.slice(1))],
    timestamp_s: Number((prevTimestamp + 0.05).toFixed(3))
  };
}

export function mockSessionFeatures(): SessionFeatureVector {
  return {
    erd_mu_c3_mean: -0.3 + Math.random() * 0.2,
    erd_mu_c4_mean: -0.28 + Math.random() * 0.2,
    erd_lat_mu_mean: -0.04 + Math.random() * 0.08,
    erd_lat_mu_abs_mean: 0.02 + Math.random() * 0.1,
    lda_conf_mean: 0.55 + Math.random() * 0.35,
    lda_margin_mean: 0.1 + Math.random() * 0.3,
    lda_margin_std: 0.02 + Math.random() * 0.08,
    mu_ratio_mean: 0.7 + Math.random() * 0.5,
    motor_relative_mean: 0.4 + Math.random() * 0.4,
    frob_dist_to_baseline: 0.2 + Math.random() * 0.5,
    intertrial_cov_var: 0.03 + Math.random() * 0.09,
    lda_session_acc: 0.55 + Math.random() * 0.35,
    p2p_max_mean: 60 + Math.random() * 80
  };
}

export function mockAnalyzeSession(features: SessionFeatureVector): SessionAnalyzeResponse {
  const normalized =
    (features.lda_session_acc + features.lda_conf_mean) / 2 -
    features.frob_dist_to_baseline * 0.25;
  const predicted = Math.max(0, Math.min(1, normalized));
  const cluster = predicted > 0.75 ? 2 : predicted > 0.5 ? 1 : 0;

  return {
    cluster,
    predicted_r: Number(predicted.toFixed(3)),
    silhouette: Number((0.3 + Math.random() * 0.45).toFixed(3))
  };
}

export function buildLayer6Series(shape: "linear" | "plateau" | "relapse") {
  return Array.from({ length: 12 }, (_, i) => {
    const session = i + 1;
    const x = i / 11;
    let r = 1 - x;
    if (shape === "plateau") r = 1 - Math.min(1, x * 0.7);
    if (shape === "relapse") r = x < 0.6 ? 1 - x * 0.7 : 0.58 + (x - 0.6) * 0.6;

    const ldaAcc = Math.max(0.35, 0.56 + r * 0.34 + (Math.random() - 0.5) * 0.04);
    const erdDrift = Math.max(0, (1 - r) * 0.8 + (Math.random() - 0.5) * 0.06);

    return {
      session,
      r: Number(r.toFixed(3)),
      ldaAcc: Number(ldaAcc.toFixed(3)),
      erdDrift: Number(erdDrift.toFixed(3))
    };
  });
}
