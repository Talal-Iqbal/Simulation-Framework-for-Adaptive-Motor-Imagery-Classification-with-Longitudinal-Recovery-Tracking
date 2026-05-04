# NeuroDrift — End-to-End MLOps Pipeline (Healthcare / BCI)

[![ci](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![train](https://github.com/OWNER/REPO/actions/workflows/train.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/train.yml)

**Full runbook (first-time + GitHub + Docker + CI/CD):** [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md)

NeuroDrift is a production-grade MLOps system around a 3-stage motor-imagery
EEG (BNCI2014_001) pipeline:

- **Stage 1** — frozen multi-subject acceptance model (Logistic Regression on
  8 raw EEG quality features).
- **Stage 2** — per-subject calibration with subject-adaptive acceptance
  gating, adaptive stopping, and CSP+LDA classification.
- **Stage 3** — longitudinal Ridge regression on `r` (recovery factor),
  KMeans session clustering, and PCA dimensionality reduction.

## Domain

**Healthcare** — clinically-aware Brain–Computer Interface monitoring.

## ML tasks (rubric coverage)

| Task                       | Where                                    |
|----------------------------|-------------------------------------------|
| Binary classification      | `src/neurodrift/models/acceptance.py`     |
| Per-subject classification | `src/neurodrift/models/classifier.py`     |
| Regression                 | `src/neurodrift/models/regressor.py`      |
| Dimensionality reduction   | PCA + CSP                                  |
| Clustering                 | `src/neurodrift/models/clustering.py`     |
| Time-series / spectral     | `feature_extractor.py` Welch PSD          |

## Repo layout

```text
src/neurodrift/        # importable package
  api/                 # FastAPI app + routers
  data/                # MOABB ingest, splits, synthetic fixtures
  features/            # FeatureExtractor / RawQualityExtractor adapter
  flows/               # Prefect flow + tasks + email notifier
  inference/           # pseudo_online_engine + TrialResult
  models/              # acceptance, classifier, regressor, clustering
  observability/       # structlog config + Prometheus metrics
  registry/            # joblib + manifest.json local registry
  testing/             # DeepChecks data + model + drift suites

tests/                 # pytest with synthetic fixtures
docker/                # Dockerfile.api, Dockerfile.prefect
.github/workflows/     # ci.yml + train.yml
scripts/               # train_all.py, seed_artifacts.py
reports/               # architecture.md, methodology.md
notebooks/1.ipynb      # original research notebook (unchanged)
feature_extractor.py   # provided
corruption_engine.py   # provided
degradation_model.py   # provided
```

## Quickstart (local dev)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .

python scripts/seed_artifacts.py
uvicorn neurodrift.api.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the OpenAPI UI.

## Run on real MOABB data

```bash
python scripts/train_all.py --global-subjects 1,3,4,5,6,7,8,9 --held-out-subject 2
```

The first run will download ~3–5 GB of EEG into `~/mne_data`. Subsequent
runs reuse the cache.

## Run with Docker Compose

```bash
docker compose build
docker compose up -d
```

Services:

- API: <http://localhost:8000>
- Prefect server UI: <http://localhost:4200>

Trigger a training run:

```bash
docker compose exec prefect-worker python scripts/train_all.py
```

## Tests

```bash
pytest -q                       # synthetic-only, no MOABB download
pytest -q -m "not slow"         # CI default
```

## API endpoints

| Method | Path                  | Description                                    |
|--------|-----------------------|------------------------------------------------|
| GET    | /health               | liveness + loaded model versions               |
| POST   | /predict/trial        | single-trial JSON prediction (numeric input)   |
| POST   | /predict/upload       | batch .npz multipart upload                    |
| POST   | /calibrate/subject    | run Stage-2 calibration; writes registry entry |
| POST   | /analyze/session      | session feature vector → cluster id + `r̂`     |
| GET    | /metrics              | Prometheus text counters                       |

## Notifications

Set the `SMTP_*` and `NOTIFY_*` environment variables (see `.env.example`).
The Prefect flow emails a summary on every terminal state — success, gate
failure, or uncaught exception. Missing SMTP secrets log a warning and skip
silently rather than blocking the pipeline.

## Reports & guides

- `docs/PIPELINE_GUIDE.md` — end-to-end pipeline: local/Docker, Prefect,
  tests, what to push to GitHub, Actions workflows.
- `reports/architecture.md` — mermaid system, CI/CD, methodology, container,
  inference-flow diagrams.
- `reports/methodology.md` — experiments, observations, limitations, future
  work.

## License

MIT.
