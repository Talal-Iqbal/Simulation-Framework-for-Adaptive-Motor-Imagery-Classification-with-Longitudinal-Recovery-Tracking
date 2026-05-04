"""Layer 6 — controlled degradation / recovery model.

Generates EEG epochs whose signal quality is governed by a known scalar
parameter ``r`` in ``[0, 1]`` (1.0 = pristine signal, 0.0 = maximally
degraded). Unlike ``corruption_engine.py`` — which samples random failure
modes for an acceptance-model — this module applies *deterministic*,
``r``-parameterized transforms at the SESSION level so every session has
a known ground-truth quality value usable for longitudinal regression
and clustering (Layer 7).

Four physically motivated transforms are composed, each acting only on
the imagery (task) window so the pre-cue rest baseline stays a clean
ERD reference:

    1) ERD amplitude scaling on motor channels   alpha in [0, 1]
    2) Lateralization mixing of C3/C4             beta  in [0, 1]
    3) Spatially correlated in-band noise         gamma >= 0
    4) Trial timing jitter (samples)              delta_ms >= 0

Two scalar-to-parameter "priors" are provided, both monotone in r and
agreeing at r=1 (pristine signal):

    prior_default — the original linear forms used throughout the notebook
        alpha(r) = 0.5 + 0.5*r, beta(r) = 0.4*(1-r),
        gamma(r) = 2.0*(1-r),  delta(r) = 50*(1-r) ms

    prior_alt — structurally different functional forms, used for
        cross-prior generalisation tests in Layer 7
        alpha(r) = 0.4 + 0.6*r^1.5         (concave ERD recovery)
        beta(r)  = 0.5*(1-r)^2             (quadratic lateralization)
        gamma(r) = 2.0*(1-r)^0.5           (sqrt noise growth)
        delta(r) = 80*(1-r)^0.7 ms         (sub-linear jitter)

Trajectories define how ``r`` evolves across a synthetic longitudinal
series of sessions:

    linear:    r(s) = s/(S-1)
    plateau:   r(s) = 1 - exp(-lambda*s), normalized to end at 1.0
    relapse:   linear baseline minus a Gaussian dip in the middle

Defaults match BNCI2014_001 (22-channel montage, 250 Hz, 2 s baseline +
4 s task) so the module drops directly into the project notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(x: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] <= padlen:
        b, a = butter(2, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


# ---------------------------------------------------------------------------
# Trajectory generators — return r value per session
# ---------------------------------------------------------------------------

def trajectory_linear(n_sessions: int) -> np.ndarray:
    """Steady linear improvement from r=0 to r=1 across n_sessions."""
    if n_sessions <= 1:
        return np.array([1.0])
    return np.linspace(0.0, 1.0, n_sessions)


def trajectory_plateau(n_sessions: int, lam: float = 0.6) -> np.ndarray:
    """Rapid early gain that asymptotes — r(s) = 1 - exp(-lambda*s),
    rescaled so r ends exactly at 1.0."""
    if n_sessions <= 1:
        return np.array([1.0])
    s = np.arange(n_sessions)
    r = 1.0 - np.exp(-lam * s)
    if r.max() > 0:
        r = r / r.max()
    return r


def trajectory_relapse(
    n_sessions: int,
    dip_center: float = 0.5,
    dip_width: float = 0.18,
    dip_depth: float = 0.5,
) -> np.ndarray:
    """Linear baseline recovery with a Gaussian-shaped dip in the middle —
    simulates a setback then re-recovery."""
    if n_sessions <= 1:
        return np.array([1.0])
    t = np.linspace(0.0, 1.0, n_sessions)
    base = t
    dip = np.exp(-((t - dip_center) ** 2) / (2 * dip_width ** 2)) * dip_depth
    return np.clip(base - dip, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Parameter-explicit transforms (operate on the task slice only)
# Each transform takes its physical parameter directly so callers can sample
# the four mechanisms independently rather than tying them to a single r.
# ---------------------------------------------------------------------------

def scale_erd(
    epoch: np.ndarray,
    alpha: float,
    sfreq: float,
    motor_ch: Sequence[int],
) -> np.ndarray:
    """Scale mu+beta band amplitude on motor channels by ``alpha`` in [0, 1].
    alpha=1 leaves the signal unchanged; alpha=0 zeroes the mu+beta
    component, removing class-discriminative ERD entirely."""
    out = epoch.copy()
    a = float(alpha)
    for ch in motor_ch:
        if ch < out.shape[0]:
            mu = _bandpass(out[ch], sfreq, 8, 13)
            beta = _bandpass(out[ch], sfreq, 13, 30)
            other = out[ch] - mu - beta
            out[ch] = other + a * mu + a * beta
    return out


def mix_lateralization(
    epoch: np.ndarray,
    beta: float,
    c3_idx: int,
    c4_idx: int,
) -> np.ndarray:
    """Blend C3 and C4 toward their mean by mix fraction ``beta`` in [0, 1].
    beta=0 leaves the contralateral contrast intact; beta=1 fully averages
    C3 and C4 (no lateralization)."""
    out = epoch.copy()
    if c3_idx < out.shape[0] and c4_idx < out.shape[0]:
        mix = float(beta)
        avg = 0.5 * (out[c3_idx] + out[c4_idx])
        out[c3_idx] = (1.0 - mix) * out[c3_idx] + mix * avg
        out[c4_idx] = (1.0 - mix) * out[c4_idx] + mix * avg
    return out


def add_correlated_noise(
    epoch: np.ndarray,
    gamma: float,
    sfreq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Spatially correlated (low-rank) in-band noise.
    ``gamma`` is a relative RMS budget (>= 0); the additive scale is further
    damped by 0.25 so gamma=2 produces ~0.5 * signal_RMS noise (clearly
    detectable but not destructive)."""
    g = float(gamma)
    if g <= 0:
        return epoch
    n_ch, n_t = epoch.shape
    n_comp = 3
    spatial = rng.standard_normal((n_ch, n_comp))
    spatial /= np.linalg.norm(spatial, axis=0, keepdims=True) + 1e-12
    temporal = rng.standard_normal((n_comp, n_t))
    temporal = _bandpass(temporal, sfreq, 4, 30)
    noise = spatial @ temporal
    rms_signal = float(np.sqrt(np.mean(epoch ** 2) + 1e-24))
    rms_noise = float(np.sqrt(np.mean(noise ** 2) + 1e-24))
    noise = noise / (rms_noise + 1e-12) * (rms_signal * g * 0.25)
    return epoch + noise


def apply_timing_jitter(
    epoch: np.ndarray,
    delta_ms: float,
    sfreq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Roll the task window by up to +/- ``delta_ms`` ms in either direction
    to simulate inconsistent MI engagement onset."""
    delta = float(delta_ms)
    if delta <= 0:
        return epoch
    max_shift = int(round(delta * 1e-3 * sfreq))
    if max_shift <= 0:
        return epoch
    shift = int(rng.integers(-max_shift, max_shift + 1))
    return np.roll(epoch, shift, axis=-1)


# ---------------------------------------------------------------------------
# Priors: deterministic mappings r -> (alpha, beta, gamma, delta_ms)
# A "Prior" is a callable that converts a scalar quality r into the four
# physical parameters. Swapping priors between train and test is the basis
# of the cross-prior generalisation test in Layer 7.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DegradationParams:
    """Physical degradation parameters consumed by the four transforms."""

    alpha: float       # ERD amplitude scale, in [0, 1]
    beta: float        # lateralization mix fraction, in [0, 1]
    gamma: float       # relative noise RMS budget, >= 0
    delta_ms: float    # max timing jitter (ms), >= 0


Prior = Callable[[float], DegradationParams]


def prior_default(r: float) -> DegradationParams:
    """Original Layer-6 functional forms (linear in r)."""
    r = float(r)
    return DegradationParams(
        alpha=0.5 + 0.5 * r,
        beta=0.4 * (1.0 - r),
        gamma=2.0 * (1.0 - r),
        delta_ms=50.0 * (1.0 - r),
    )


def prior_alt(r: float) -> DegradationParams:
    """Alternative prior with structurally different functional forms,
    monotone in r and agreeing with prior_default at r=1.

    Used for cross-prior generalisation tests: a regressor trained against
    prior_default should *fail* to predict r/parameters when sessions are
    generated from prior_alt unless it has learned mechanism-level features
    rather than the specific equations. The drop in R^2 is the real
    generalisation number for the framework.
    """
    r = float(r)
    one_minus = 1.0 - r
    return DegradationParams(
        alpha=0.4 + 0.6 * (r ** 1.5),
        beta=0.5 * one_minus ** 2,
        gamma=2.0 * one_minus ** 0.5,
        delta_ms=80.0 * one_minus ** 0.7,
    )


def r_to_params(r: float, prior: Optional[Prior] = None) -> DegradationParams:
    """Map a scalar quality value to physical parameters via the chosen
    prior. Defaults to ``prior_default`` to preserve original behaviour."""
    p = prior or prior_default
    return p(r)


def sample_params_uniform(
    rng: np.random.Generator,
    alpha_range: Tuple[float, float] = (0.4, 1.0),
    beta_range:  Tuple[float, float] = (0.0, 0.5),
    gamma_range: Tuple[float, float] = (0.0, 2.5),
    delta_range: Tuple[float, float] = (0.0, 60.0),
) -> DegradationParams:
    """Sample (alpha, beta, gamma, delta_ms) independently from uniform
    ranges that cover the same physical envelope as ``prior_default`` over
    r in [0, 1]. Useful for breaking the 1-D r manifold so a regressor must
    learn each mechanism separately rather than memorising a single curve.
    """
    return DegradationParams(
        alpha=float(rng.uniform(*alpha_range)),
        beta=float(rng.uniform(*beta_range)),
        gamma=float(rng.uniform(*gamma_range)),
        delta_ms=float(rng.uniform(*delta_range)),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

@dataclass
class DegradationModel:
    """Apply r-parameterized degradation to BNCI2014_001-style epochs.

    Parameters mirror the project notebook: 250 Hz sampling, a 2 s pre-cue
    baseline followed by a 4 s imagery window (BASELINE_SAMPLES = 500).
    Channel indices C3=7, Cz=9, C4=11 follow the 22-channel montage used
    by ``corruption_engine.py``.

    The instance carries a default ``prior`` (``prior_default``) but every
    r-based method accepts a per-call ``prior`` override so a single model
    can serve both in-distribution and cross-prior evaluation passes.
    """

    sfreq: float = 250.0
    baseline_samples: int = 500
    c3_idx: int = 7
    cz_idx: int = 9
    c4_idx: int = 11
    seed: int = 0
    prior: Optional[Prior] = field(default=None)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.motor_ch: Tuple[int, int, int] = (self.c3_idx, self.cz_idx, self.c4_idx)
        if self.prior is None:
            self.prior = prior_default

    # ------------------------------------------------------------------
    # parameter-explicit path: callers pass (alpha, beta, gamma, delta_ms)
    # directly. Bypasses r and any prior — required for multi-output
    # regression with independently sampled parameters.
    # ------------------------------------------------------------------

    def degrade_trial_params(
        self,
        epoch: np.ndarray,
        params: DegradationParams,
        jitter: bool = True,
    ) -> np.ndarray:
        """Apply the four transforms with explicit physical parameters,
        acting only on the imagery window."""
        full = epoch.copy()
        task = full[..., self.baseline_samples:].copy()
        task = scale_erd(task, params.alpha, self.sfreq, self.motor_ch)
        task = mix_lateralization(task, params.beta, self.c3_idx, self.c4_idx)
        task = add_correlated_noise(task, params.gamma, self.sfreq, self.rng)
        if jitter and params.delta_ms > 0:
            task = apply_timing_jitter(task, params.delta_ms, self.sfreq, self.rng)
        full[..., self.baseline_samples:] = task
        return full

    def degrade_session_params(
        self,
        X: np.ndarray,
        params: DegradationParams,
        jitter: bool = True,
    ) -> np.ndarray:
        """Apply the same explicit parameters to every trial in a session."""
        out = np.empty_like(X)
        for i in range(len(X)):
            out[i] = self.degrade_trial_params(X[i], params, jitter=jitter)
        return out

    # ------------------------------------------------------------------
    # r-based path (back-compat): callers pass a scalar r and an optional
    # prior; the prior maps r to the four parameters and the parameter-
    # explicit path executes the rest.
    # ------------------------------------------------------------------

    def degrade_trial(
        self,
        epoch: np.ndarray,
        r: float,
        jitter: bool = True,
        prior: Optional[Prior] = None,
    ) -> np.ndarray:
        """Apply all four r-parameterized transforms to one trial."""
        params = r_to_params(r, prior=prior or self.prior)
        return self.degrade_trial_params(epoch, params, jitter=jitter)

    def degrade_session(
        self,
        X: np.ndarray,
        r: float,
        jitter: bool = True,
        prior: Optional[Prior] = None,
    ) -> np.ndarray:
        """Apply the same r to every trial in a session."""
        params = r_to_params(r, prior=prior or self.prior)
        return self.degrade_session_params(X, params, jitter=jitter)

    def build_trajectory(self, n_sessions: int, shape: str = "linear", **kwargs) -> np.ndarray:
        if shape == "linear":
            return trajectory_linear(n_sessions)
        if shape == "plateau":
            return trajectory_plateau(n_sessions, **kwargs)
        if shape == "relapse":
            return trajectory_relapse(n_sessions, **kwargs)
        raise ValueError(f"Unknown trajectory shape: {shape!r}")

    def generate_longitudinal(
        self,
        X_clean: np.ndarray,
        n_sessions: int,
        shape: str = "linear",
        jitter: bool = True,
        prior: Optional[Prior] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, List[Tuple[float, np.ndarray]]]:
        """Build a synthetic longitudinal series of ``n_sessions`` sessions
        from a single clean source session ``X_clean``.

        Parameters
        ----------
        prior : optional Prior
            Override the model's default r-to-parameters mapping for this
            run (e.g. pass ``prior_alt`` to generate a held-out test set
            for cross-prior generalisation evaluation).

        Returns
        -------
        rs : np.ndarray
            r value for each session (length n_sessions).
        sessions : list of (r, X_session)
            Per-session degraded epoch tensors, in trajectory order.
        """
        rs = self.build_trajectory(n_sessions, shape=shape, **kwargs)
        sessions: List[Tuple[float, np.ndarray]] = []
        for r in rs:
            X_deg = self.degrade_session(X_clean, float(r), jitter=jitter, prior=prior)
            sessions.append((float(r), X_deg))
        return rs, sessions
