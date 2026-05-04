# NeuroDrift — System Architecture

## 1. Domain & Problem Statement

**Domain:** Healthcare — specifically, motor-imagery Brain–Computer Interface (BCI)
recovery monitoring. Patients with motor impairment train an EEG-decoded
classifier of left-hand vs right-hand imagery. Day-to-day, the signal quality
of an EEG session varies (electrode contact, fatigue, neurological state). A
quality-aware system that gates low-quality trials at inference time and
tracks quality drift across sessions is therefore clinically useful.

**ML Tasks Covered (rubric requirement):**

| Task                       | Module                                    |
|----------------------------|-------------------------------------------|
| Binary classification      | `models/acceptance.py` (LogReg)            |
| Per-subject classification | `models/classifier.py` (CSP + LDA)         |
| Regression                 | `models/regressor.py` (Ridge for `r`)      |
| Dimensionality reduction   | PCA in `models/regressor.py`, CSP overall  |
| Clustering                 | `models/clustering.py` (KMeans)            |
| Time-series / spectral     | Welch PSD + ERD inside `feature_extractor` |

## 2. Three-Stage Pipeline (mirrors the research notebook)

```mermaid
flowchart LR
    subgraph Stage1[Stage 1 - Global Acceptance Model FROZEN]
        DataIngest[MOABB ingest subjects 1,3-9] --> RQE[RawQualityExtractor 8 features]
        RQE --> Corrupt[CorruptionEngine hard negatives]
        Corrupt --> Build[build_acceptance_dataset]
        Build --> TrainAcc[Train LogisticRegression]
    end

    subgraph Stage2[Stage 2 - Per-subject Calibration]
        HoldOut[Held-out subject calibration pool] --> Gate[Acceptance gate top-70 percentile]
        Gate --> Adaptive[Adaptive stop CSP+LDA validation]
        Adaptive --> CalibFit[Fit CSP n=4 + LDA]
    end

    subgraph Stage3[Stage 3 - Longitudinal Layer 6 + 7]
        SessionVec[session_feature_vector 13 features] --> Ridge[Ridge regression - r LOO]
        SessionVec --> KMeans[KMeans n=3]
    end

    TrainAcc -.frozen.-> Gate
    CalibFit --> SessionVec
```

## 3. End-to-End MLOps Topology

```mermaid
flowchart TB
    subgraph DevSide[Developer / CI]
        GH[GitHub repo] --> CI[GitHub Actions ci.yml]
        GH --> Train[GitHub Actions train.yml]
    end

    subgraph Orchestration[Prefect Orchestration]
        Flow[neurodrift_training_flow]
        Tasks[Ingest -> Features -> DeepChecks -> Train -> Calibrate -> Layer7 -> Save]
    end

    subgraph Artifacts[Local Model Registry]
        Reg[(artifacts/registry)]
        Acc[acceptance_model versioned]
        Cal[calibration_subject_X versioned]
        L7[session_r_regressor + session_kmeans]
    end

    subgraph Serving[FastAPI Service]
        Health[/health/]
        Predict[/predict/trial /predict/upload/]
        Calibrate[/calibrate/subject/]
        Analyze[/analyze/session/]
        Metrics[/metrics/]
    end

    subgraph Notify[Notifications]
        Email[SMTP email NOTIFY_TO]
    end

    GH --> Flow
    Train --> Flow
    Flow --> Tasks
    Tasks --> Reg
    Reg --> Acc
    Reg --> Cal
    Reg --> L7
    Reg --> Serving
    Tasks -- on success / failure --> Email
    CI --> Reg
```

## 4. Data Flow per Inference Request

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant RQE as RawQualityExtractor
    participant Acc as AcceptanceModel (frozen)
    participant CSP as CSP + LDA (subject)

    Client->>API: POST /predict/trial {epoch, y_true?}
    API->>RQE: extract(epoch) -> 8d features
    RQE->>Acc: scaler + LR predict_proba
    Acc-->>API: P(accept)
    alt P(accept) >= cal_threshold
        API->>CSP: transform(task_view) + LDA
        CSP-->>API: y_pred, confidence, margin
        API-->>Client: TrialResult accepted=true
    else
        API->>API: derive rejection reasons
        API-->>Client: TrialResult accepted=false reasons=[...]
    end
```

## 5. Containerization Layout

```mermaid
flowchart LR
    Compose[docker-compose.yml] --> APIc[api - Dockerfile.api uvicorn]
    Compose --> Server[prefect-server - prefect/prefect:2.18]
    Compose --> Worker[prefect-worker - Dockerfile.prefect]
    APIc --> SharedVol[(artifacts shared volume)]
    Worker --> SharedVol
    Worker --> MOABBVol[(moabb_data volume)]
```

- API: multi-stage `python:3.11-slim`, non-root user `neurodrift`,
  `HEALTHCHECK` curls `/health`.
- Prefect Server: official image, exposes UI on `:4200`.
- Prefect Worker: same Python base + project source; runs the flow on demand.
- Volumes: `artifacts` (registry shared between API and worker so the API
  always sees the latest models written by training).

## 6. CI/CD Pipeline Explanation

```mermaid
flowchart LR
    Push[git push / PR] --> CI[ci.yml]
    CI --> Lint[ruff + mypy]
    CI --> Tests[pytest synthetic fixtures]
    CI --> Suite[DeepChecks smoke checks]
    CI --> SeedSmoke[seed_artifacts synthetic]
    CI --> DockerBuild[docker build api image no push]

    Manual[workflow_dispatch / cron] --> TrainWF[train.yml]
    TrainWF --> CacheMOABB[Cache ~/mne_data]
    TrainWF --> RunFlow[scripts/train_all.py]
    RunFlow --> RealDC[DeepChecks suites]
    RunFlow --> RegistryW[artifacts/registry]
    TrainWF --> UploadArt[Upload registry artifact]
    TrainWF --> GHCR[Push image to ghcr.io]
```

- **CI (`ci.yml`)** runs on every push/PR. Uses synthetic EEG only — no MOABB
  download — so the workflow finishes in under five minutes.
- **Training (`train.yml`)** runs on `workflow_dispatch` or weekly cron. Caches
  `~/mne_data` between runs, executes the full Prefect flow against real
  BNCI2014_001 data, runs DeepChecks gates, and on success uploads model
  artifacts and pushes a versioned image to GHCR.

## 7. Prefect Orchestration Detail

Each task carries `retries=3, retry_delay_seconds=30` to absorb transient
MOABB / network failures. Tasks delegate to pure functions in
`src/neurodrift/{data,models,testing,registry}` so the same code runs from
the FastAPI app, from `pytest`, and from CI.

The flow's `try/except` envelope sends an email summary on every terminal
state via `flows/notify.py` (uses `smtplib.SMTP` with `STARTTLS`). The
notifier is non-fatal: a missing SMTP secret logs a warning rather than
failing the run.

## 8. Methodology Flow Diagram

```mermaid
flowchart TB
    Start([Research Notebook 1.ipynb]) --> Refactor[Refactor into src/neurodrift package]
    Refactor --> Tests[Add pytest fixtures + DeepChecks suites]
    Tests --> Wrap[Wrap behind FastAPI + Pydantic schemas]
    Wrap --> Orchestrate[Wrap pipeline in Prefect tasks]
    Orchestrate --> Containerize[Dockerise API + worker]
    Containerize --> CICD[GitHub Actions: CI + train workflows]
    CICD --> Notify[Email notifications on success/failure]
    Notify --> Drift[Drift detection PSI / KS for retraining trigger]
    Drift --> Done([Production-ready MLOps system])
```

## 9. Out of Scope Items

- Online retraining from production prediction outcomes — the acceptance
  model is **frozen** by design, exactly as the notebook specifies.
- GPU / deep-learning models — the notebook stack stays classical.
- Production secret manager — environment variables only, with a clear
  `.env.example`.
