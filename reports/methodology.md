# NeuroDrift — Methodology, Experiments, and Observations

## 1. Introduction

NeuroDrift is an end-to-end MLOps pipeline for motor-imagery EEG, treating a
single notebook prototype (`notebooks/1.ipynb`) as a production-grade ML
system. The chosen domain is **Healthcare** — Brain–Computer Interface
recovery monitoring. The pipeline meets the rubric's requirement of multiple
ML tasks in one workflow:

1. **Binary classification** — global acceptance gate (Logistic Regression)
2. **Per-subject classification** — CSP + LDA on motor-imagery trials
3. **Regression** — Ridge regression on session-level features predicting `r`
4. **Dimensionality reduction** — CSP + PCA
5. **Clustering** — KMeans on session feature vectors
6. **Time-series / spectral** — Welch PSD, ERD, fatigue & engagement indices

## 2. Problem Statement

Standard MI-EEG pipelines train per-subject CSP+LDA on calibration data
without quality control. In real clinics, electrode pop, EOG residual,
fatigue, and disengagement contaminate trials — the classifier still emits
a high-confidence prediction even when the trial carries no usable
discriminative signal. NeuroDrift adds a frozen, multi-subject **acceptance
gate** that decides per trial whether to forward the trial to the calibrated
classifier or reject it with a structured reason flag.

## 3. ML Experiments & Comparison

### 3.1 Acceptance model (Stage 1)

Trained on subjects {1, 3, 4, 5, 6, 7, 8, 9} of BNCI2014_001 (1152 clean
trials) plus 2× CorruptionEngine outputs (2304 hard negatives) plus 106
borderline negatives (clean trials with weak ERD lateralization).

| Model variant                                  | Test AUC | Test Acc |
|------------------------------------------------|----------|----------|
| Logistic Regression (baseline, balanced)       | **0.711** | 0.62     |
| LR without borderline negatives (notebook ref) | 0.69     | 0.60     |
| LR with `class_weight=None`                    | 0.69     | 0.71*    |

*Higher accuracy but precision on the minority class collapses; balanced
weighting kept for clinical recall.

Top standardized coefficients (from notebook `cell-global-model-eval`):

```
motor_relative_power      coef=+0.622
kurtosis_max              coef=-0.489
peak_to_peak_max          coef=-0.283
baseline_p2p_max          coef=-0.214
mu_ratio_motor            coef=+0.141
```

### 3.2 Per-subject calibration (Stage 2)

Subject 2, top-70% acceptance gate (subject-adaptive percentile rule):

| Setting                              | Cal trials | In-sample acc | Eval acc |
|--------------------------------------|------------|----------------|---------|
| Adaptive stop, CSP n=4, accept 70%   | 84         | **0.86**       | 0.56    |
| Fixed 132 trials, no gate (notebook) | 132        | 0.83           | 0.55    |
| Adaptive stop, CSP n=6               | 84         | 0.85           | 0.55    |

Adaptive stopping picks 84/144 trials. Cross-session generalisation gap is
the well-known BCI cross-session drift; the gate concentrates rejections on
hard-to-classify trials so net accuracy on the *accepted* sub-population is
typically better than on the full set.

### 3.3 Layer-7 longitudinal regression

Ridge regression on 13 session-level features predicting the latent
recovery factor `r` ∈ [0, 1] (LOO CV across 18 sessions in 3 trajectory
shapes — linear / plateau / relapse):

| Setting                                  | R² (LOO) | RMSE  |
|------------------------------------------|----------|-------|
| Ridge α=1 on raw 13-d feature vector     | **0.94** | 0.10  |
| Ridge α=1 + PCA(5)                       | 0.92     | 0.11  |
| Ridge cross-prior (default→alt)          | 0.74     | 0.20  |

The cross-prior drop quantifies the model's failure to generalise across
different functional forms of `r → params` and is the meaningful
generalisation number for the framework.

### 3.4 KMeans session clustering

`n_clusters=3` on the standardized session panel produces clusters that map
intuitively to (low-r impaired / mid-r recovering / high-r near-pristine).
Silhouette ≈ 0.45 on the synthetic 18-session panel.

## 4. Final Observations

- **Best-performing model**: balanced Logistic Regression (acceptance) +
  CSP n=4 + LDA (Stage 2). Replacing Ridge with multi-output Ridge for the
  four physical params (alpha, beta, gamma, delta_ms) gave per-mechanism
  R² in [0.6, 0.9].
- **Data quality issues**: the 95th-percentile p2p threshold flagged ~6% of
  clean trials; CorruptionEngine produced realistic gradients (mild → severe)
  that allowed the acceptance gate to learn a sharp decision boundary
  rather than a degenerate "all-clean / all-corrupted" one.
- **Overfitting / underfitting**: Stage 2 in-sample accuracy of 0.86 vs.
  eval accuracy of 0.56 is the classic BCI cross-session generalisation
  gap, not a refactor regression — the same gap was present in the notebook.
- **Deployment speed via CI/CD**: the `ci.yml` job (synthetic data only)
  finishes in ~5 minutes. Adding DeepChecks + Docker build keeps wall-time
  under 7 minutes. Real-data retraining (`train.yml`) takes ~15–25 minutes
  thanks to the cached `~/mne_data` MOABB downloads.
- **Reliability via Prefect**: each task has `retries=3, retry_delay=30s`,
  so transient MOABB outages no longer abort the entire flow. The
  `try/except` envelope guarantees an email even on uncaught exceptions.

## 5. Limitations

- Acceptance model is frozen — drift on a new clinical population requires a
  Prefect-orchestrated retraining run, not online updates.
- Synthetic CI fixtures are deliberately uninformative for ML accuracy; the
  CI suite proves only contract / shape correctness.
- Email notifications use plain SMTP. In a real deployment, swap for a
  secure transactional API (SendGrid, SES) and a proper secret manager.
- Layer-7 R² values are computed on synthetic longitudinal panels seeded
  from `DegradationModel`. Real longitudinal cohorts will have noisier `r`
  values.

## 6. Future Work

- Add experiment tracking (MLflow or Weights & Biases) instead of the local
  joblib + manifest registry.
- Replace local file-based registry with an S3-compatible object store and
  a Postgres metadata DB.
- Add an HTTP / WebSocket streaming endpoint so the FastAPI app can drive
  real-time BCI sessions instead of batch `.npz` uploads.
- Online drift monitoring on the production prediction stream (the current
  PSI / KS module is offline; wire it to a Prometheus exporter and set
  alerts on the Grafana dashboard).
- Integrate a concept-drift retraining trigger that calls
  `prefect deployment run` automatically once PSI exceeds a threshold.
