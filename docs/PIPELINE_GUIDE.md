# NeuroDrift — Complete Pipeline Guide

This document walks through **first-time setup**, **ongoing use**, and **GitHub / CI/CD** so you can run the full MLOps stack **without the frontend** and align with course-style requirements (FastAPI, Prefect, Docker, DeepChecks, Actions).

---

## 1. What “the pipeline” includes

| Stage | What happens |
|-------|----------------|
| **Training (Prefect flow)** | MOABB ingest → feature extraction → corruption / dataset → **DeepChecks** data suite → train acceptance model → **DeepChecks** model suite → Stage-2 calibration → Layer-7 regression/clustering → **versioned saves** to local registry → optional **email** notification |
| **Serving (FastAPI)** | Load registry artifacts → `/predict/trial`, `/predict/upload`, `/health`, `/metrics`, etc. |
| **CI (GitHub Actions)** | Lint, mypy, pytest (synthetic), `seed_artifacts` smoke, **Docker build** (no push) |
| **Train + CD workflow** | Full `train_all.py` on MOABB, upload **artifacts**, **push API image to GHCR** |

Paths that matter locally:

- **Registry (ignored by git):** `artifacts/registry/` — created by `seed_artifacts.py` or `train_all.py`.
- **MOABB cache:** `~/mne_data` (or `%USERPROFILE%\mne_data` on Windows) — large; not committed.

---

## 2. Prerequisites

- **Python 3.10 or 3.11** (repo targets `<3.12`; CI uses 3.11).
- **Git**.
- **Docker Desktop** (optional but required for containerization demo / Compose).
- **Disk:** ~5 GB free if you run **full MOABB** training the first time.
- **GitHub account** (for remote CI/CD).

---

## 3. First-time setup (local, every machine)

### 3.1 Clone and virtual environment

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (cmd):
.venv\Scripts\activate
```

### 3.2 Install the project

```bash
python -m pip install --upgrade pip wheel
pip install -r requirements-dev.txt
pip install -e .
```

### 3.3 Environment variables

Copy the template and edit **only** what you need:

```bash
cp .env.example .env
```

- For **local API + default registry paths**, defaults in `.env.example` are enough.
- For **training email notifications**, set `SMTP_*` and `NOTIFY_*`. If omitted, training still completes; the notifier logs a warning.

Never commit `.env` (it is gitignored).

---

## 4. Two ways to produce models

### Path A — Fast demo / CI parity (~minutes, no MOABB download)

Use synthetic fixtures written into `artifacts/registry/`:

```bash
python scripts/seed_artifacts.py
```

Then start the API (section 5). This matches what **`ci.yml`** exercises (`seed_artifacts` smoke).

### Path B — Full course pipeline (~15–45+ minutes first run, real BNCI2014_001)

Runs the complete **`neurodrift_training_flow`** (Prefect tasks: ingest → … → DeepChecks → train → save):

```bash
python scripts/train_all.py --global-subjects 1,3,4,5,6,7,8,9 --held-out-subject 2
```

First run downloads MOABB/MNE data into **`~/mne_data`** (cached afterward).

---

## 5. Run the FastAPI service (after Path A or B)

```bash
uvicorn neurodrift.api.main:app --reload --host 0.0.0.0 --port 8000
```

- **OpenAPI UI:** http://localhost:8000/docs  
- **Health:** http://localhost:8000/health  

Try predictions only after `/health` shows the expected registry entries (e.g. seeded or trained calibration for your subject).

---

## 6. Automated tests (local)

Same spirit as CI:

```bash
pytest tests/ -m "not slow" -q
```

Full suite (may include slower tests):

```bash
pytest tests/ -q
```

---

## 7. Docker & Docker Compose

### 7.1 First time with Compose

Build and start API + Prefect server + worker:

```bash
docker compose build
docker compose up -d
```

Important: the **`artifacts`** volume starts **empty**. Before the API can serve real predictions you must either:

- Run training **inside** the worker (recommended for a full demo):

  ```bash
  docker compose exec prefect-worker python scripts/train_all.py \
    --global-subjects 1,3,4,5,6,7,8,9 --held-out-subject 2
  ```

  (MOABB data is cached in the **`moabb_data`** volume.)

- **Or** seed synthetic artifacts inside a container that shares the volume (e.g. a one-off run using the same image/context as the worker).

URLs:

- API: http://localhost:8000  
- Prefect UI: http://localhost:4200  

### 7.2 Later use

```bash
docker compose up -d      # start
docker compose logs -f api # logs
docker compose down        # stop
```

---

## 8. Prefect orchestration (requirements checklist)

- **Implemented:** `src/neurodrift/flows/training_flow.py` — `@flow` + `@task`, retries on ingest, failure/success **email** wrapper.
- **Run locally:** `python scripts/train_all.py` (submits a flow run; if `PREFECT_API_URL` points at a server, runs appear in the UI).
- **Compose:** Prefect server on **:4200**; worker has `PREFECT_API_URL` set — exec `train_all.py` as above.
- **Optional:** register deployments via `prefect.yaml` (`prefect deploy` / work pool) — not required if graders accept script-triggered flow runs + diagram in `reports/architecture.md`.

---

## 9. What to push to GitHub (and what never to push)

### 9.1 Push these

- All **source**: `src/neurodrift/`, `tests/`, `scripts/`
- Root modules: `feature_extractor.py`, `corruption_engine.py`, `degradation_model.py`
- **`pyproject.toml`**, `requirements.txt`, `requirements-dev.txt`
- **`docker/`**, `docker-compose.yml`, **`prefect.yaml`**
- **`.github/workflows/`** (`ci.yml`, `train.yml`)
- **`README.md`**, **`docs/`**, **`reports/*.md`**, **`notebooks/`** if part of your story
- **`.env.example`**, **`.gitignore`**

### 9.2 Never push (must stay gitignored)

- `.venv/`, `__pycache__/`, caches  
- **`.env`** (secrets)  
- **`artifacts/`**, **`data/`**, **`mne_data/`**, downloaded EEG  
- **`node_modules/`**, **`frontend/dist/`** (if present)  
- Generated **`reports/deepchecks/`**, **`reports/*.html`**

After fixing `.gitignore`, if something huge was already committed:

```bash
git rm -r --cached path/to/accidental-folder
```

---

## 10. GitHub repository setup (first time)

1. Create an empty repo on GitHub (no README required if you already have one locally).
2. Replace **`OWNER/REPO`** in `README.md` badge URLs with your **`username/repo`**.
3. Default branch: use **`main`** or **`develop`** — **`ci.yml`** triggers on both.

```bash
git init   # if not already a repo
git add .
git status # verify no artifacts, .venv, .env, node_modules
git commit -m "Initial commit: NeuroDrift MLOps pipeline"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

---

## 11. GitHub Actions — what happens automatically

### 11.1 `ci.yml` (Continuous Integration)

**Triggers:** push / PR to `main` or `develop`.

**Does:** install → **ruff** → **mypy** (non-blocking `|| true`) → **pytest** synthetic → **`seed_artifacts.py`** → **Docker build** for API (**no registry push**).

**Your job:** merge only when this workflow is green (fix lint/tests locally first).

### 11.2 `train.yml` (training + image publish)

**Triggers:**

- **Manual:** Actions → **train** → **Run workflow** (optional inputs for subject IDs).
- **Schedule:** weekly cron (Sunday 04:00 UTC).

**Does:** cache **`~/mne_data`** → **`train_all.py`** → upload **`artifacts/registry`** as workflow artifact → on success **push** `ghcr.io/<owner>/neurodrift-api:<sha>` and `:latest`.

**Permissions:** workflow sets `packages: write` for GHCR.

**Optional secrets** (for email during CI training — leave unset if not needed):

| Secret | Purpose |
|--------|---------|
| `SMTP_HOST` | SMTP server |
| `SMTP_PORT` | Port (e.g. 587) |
| `SMTP_USER` / `SMTP_PASS` | Credentials |
| `NOTIFY_FROM` / `NOTIFY_TO` | Email addresses |

---

## 12. Ongoing workflow (after first success)

| Goal | Command / action |
|------|-------------------|
| Daily coding | Feature branch → PR → wait for **ci.yml** |
| Refresh synthetic registry locally | `python scripts/seed_artifacts.py` |
| Full retrain locally | `python scripts/train_all.py` with desired flags |
| Retrain in cloud | Run **train** workflow manually on GitHub |
| Pull image built by CI | `docker pull ghcr.io/<owner>/neurodrift-api:latest` (requires GHCR auth if private) |
| Prefect UI | Compose: http://localhost:4200 |

---

## 13. Requirement mapping (quick audit)

| Requirement | Where demonstrated |
|-------------|-------------------|
| FastAPI + JSON / upload | `/predict/trial`, `/predict/upload`, `/docs` |
| Prefect pipeline | `training_flow.py` + `train_all.py` |
| DeepChecks | Training tasks + testing modules under `src/neurodrift/testing/` |
| Docker | `docker/Dockerfile.api`, `Dockerfile.prefect`, `docker-compose.yml` |
| CI/CD | `.github/workflows/ci.yml`, `train.yml` |
| Multiple ML tasks | See `README.md` table + `reports/methodology.md` |

---

## 14. Further reading

- **`reports/architecture.md`** — system, CI/CD, Prefect, container diagrams (Mermaid).
- **`reports/methodology.md`** — experiments, observations, limitations, future work.

If anything fails (MOABB download, Docker volume empty, GHCR push), check logs for that step first; **`docker compose logs`** and GitHub Actions run logs are the fastest breadcrumbs.
