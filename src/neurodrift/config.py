"""Project-wide constants and runtime configuration.

Constants are mirrored from the original NeuroDrift research notebook
(`notebooks/1.ipynb`, `cell-constants`). Paths and runtime knobs can be
overridden via environment variables (see `.env.example`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

SFREQ: float = 250.0
BASELINE_SAMPLES: int = int(2.0 * SFREQ)
TASK_SAMPLES: int = int(4.0 * SFREQ)
EPOCH_SAMPLES: int = BASELINE_SAMPLES + TASK_SAMPLES + 1
N_CHANNELS: int = 22

ITI_S: float = 1.5
FEEDBACK_DELAY_S: float = 0.2
CLASSIFICATION_DELAY_S: float = 0.05

CH_NAMES: list[str] = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

C3_IDX: int = CH_NAMES.index("C3")
CZ_IDX: int = CH_NAMES.index("Cz")
C4_IDX: int = CH_NAMES.index("C4")

KURT_THRESHOLD: float = 5.0
ERDLAT_THRESHOLD: float = 0.05

CSP_N_COMPONENTS: int = 4

ROOT_DIR: Path = Path(__file__).resolve().parents[2]


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw) if raw else default


def _env_int_list(var: str, default: list[int]) -> list[int]:
    raw = os.environ.get(var)
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    registry_dir: Path = field(
        default_factory=lambda: _env_path("NEURODRIFT_REGISTRY_DIR", ROOT_DIR / "artifacts" / "registry")
    )
    data_dir: Path = field(
        default_factory=lambda: _env_path("NEURODRIFT_DATA_DIR", ROOT_DIR / "data")
    )
    log_level: str = field(default_factory=lambda: os.environ.get("NEURODRIFT_LOG_LEVEL", "INFO"))

    api_host: str = field(default_factory=lambda: os.environ.get("NEURODRIFT_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.environ.get("NEURODRIFT_API_PORT", "8000")))

    held_out_subject: int = field(
        default_factory=lambda: int(os.environ.get("NEURODRIFT_HELD_OUT_SUBJECT", "2"))
    )
    global_train_subjects: list[int] = field(
        default_factory=lambda: _env_int_list(
            "NEURODRIFT_GLOBAL_TRAIN_SUBJECTS", [1, 3, 4, 5, 6, 7, 8, 9]
        )
    )

    smtp_host: str = field(default_factory=lambda: os.environ.get("SMTP_HOST", ""))
    smtp_port: int = field(default_factory=lambda: int(os.environ.get("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.environ.get("SMTP_USER", ""))
    smtp_pass: str = field(default_factory=lambda: os.environ.get("SMTP_PASS", ""))
    notify_from: str = field(default_factory=lambda: os.environ.get("NOTIFY_FROM", ""))
    notify_to: str = field(default_factory=lambda: os.environ.get("NOTIFY_TO", ""))

    def ensure_dirs(self) -> None:
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    return Settings()
