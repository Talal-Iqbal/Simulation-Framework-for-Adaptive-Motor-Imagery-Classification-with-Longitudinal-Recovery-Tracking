"""Corruption engine for MI-EEG trials.

Turns clean (already 8-32 Hz bandpassed) epochs into realistic "bad"
trials of five kinds:

    - artifact contamination (EMG burst, EOG residual, electrode ring)
    - subject disengagement (ERD attenuation, lateralization loss, drowsy, rest)
    - physiological implausibility (channel-shuffle, hemisphere swap)
    - SNR degradation (in-band pink noise, channel dropout)
    - borderline / mild realistic failures (weak ERD, partial lateralization,
      borderline-SNR noise) — dedicated near-threshold cases that teach the
      acceptance model a sharp decision boundary

Each primitive is a pure function
    fn(epoch, sfreq, severity, rng) -> epoch
and is agnostic to epoch length. CorruptionEngine samples a family, a
primitive within the family, and a severity bucket, then returns the
corrupted trial plus metadata describing what was done.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt


def _bandpass(x: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    # filtfilt needs enough samples; fall back to a lower order on short inputs.
    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] <= padlen:
        b, a = butter(2, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-24))


# ---------------------------------------------------------------------------
# Artifact family
# ---------------------------------------------------------------------------

def inject_emg_burst(epoch, sfreq, severity, rng, peripheral_idx=None):
    """In-band EMG: broadband noise bandpassed into 20-32 Hz, windowed,
    added to a peripheral channel."""
    n_ch, n_t = epoch.shape
    n_bursts = 1 + int(severity * 2)
    out = epoch.copy()
    min_dur = int(sfreq * 0.15)
    for _ in range(n_bursts):
        dur = int(sfreq * rng.uniform(0.15, 0.5))
        dur = max(dur, min_dur)
        start = int(rng.integers(0, max(1, n_t - dur)))
        raw = rng.standard_normal(dur)
        emg = _bandpass(raw, sfreq, 20, 32)
        envelope = np.hanning(dur)
        burst = emg * envelope
        amp = (0.5 + 2.5 * severity) * _rms(epoch)
        burst = burst / _rms(burst) * amp
        if peripheral_idx is None:
            ch = int(rng.integers(0, n_ch))
        else:
            ch = int(rng.choice(peripheral_idx))
        out[ch, start:start + dur] += burst
    return out


def inject_eog_residual(epoch, sfreq, severity, rng, frontal_idx=None):
    """Blink residual surviving 8-32 Hz bandpass: short 8-12 Hz transient
    with a frontal-to-posterior amplitude gradient."""
    n_ch, n_t = epoch.shape
    dur = int(sfreq * rng.uniform(0.15, 0.35))
    start = int(rng.integers(0, max(1, n_t - dur)))
    raw = rng.standard_normal(dur)
    blink = _bandpass(raw, sfreq, 8, 12) * np.hanning(dur)
    amp = (0.8 + 3.0 * severity) * _rms(epoch)
    blink = blink / _rms(blink) * amp

    if frontal_idx is None:
        frontal_idx = list(range(min(6, n_ch)))
    gradient = np.linspace(1.0, 0.1, n_ch)
    out = epoch.copy()
    for ch in range(n_ch):
        weight = gradient[ch] if ch in frontal_idx else 0.2 * gradient[ch]
        out[ch, start:start + dur] += weight * blink
    return out


def inject_electrode_ring(epoch, sfreq, severity, rng):
    """Damped in-band oscillation on a single channel (post-bandpass
    residual of an electrode pop)."""
    n_ch, n_t = epoch.shape
    ch = int(rng.integers(0, n_ch))
    dur = int(sfreq * rng.uniform(0.1, 0.3))
    start = int(rng.integers(0, max(1, n_t - dur)))
    t = np.arange(dur) / sfreq
    freq = rng.uniform(10, 25)
    tau = rng.uniform(0.03, 0.1)
    ring = np.sin(2 * np.pi * freq * t) * np.exp(-t / tau)
    amp = (1.0 + 4.0 * severity) * _rms(epoch)
    out = epoch.copy()
    out[ch, start:start + dur] += ring * amp
    return out


# ---------------------------------------------------------------------------
# Disengagement family
# ---------------------------------------------------------------------------

def attenuate_erd(epoch, sfreq, severity, rng, motor_ch=(7, 11)):
    """Scale mu/beta band power on motor channels toward zero so class-specific
    ERD collapses. Default indices correspond to C3=7, C4=11 in the 22-channel
    BNCI2014_001 montage used in this project."""
    out = epoch.copy()
    factor = max(0.0, 1.0 - 0.3 - 0.6 * severity)
    for ch in motor_ch:
        if ch < out.shape[0]:
            mu = _bandpass(out[ch], sfreq, 8, 13)
            beta = _bandpass(out[ch], sfreq, 13, 30)
            other = out[ch] - mu - beta
            out[ch] = other + factor * mu + factor * beta
    return out


def destroy_lateralization(epoch, sfreq, severity, rng, c3_idx=7, c4_idx=11):
    """Average C3 and C4 toward each other: preserves total motor-cortex power
    but erases the left/right contrast the LDA depends on."""
    out = epoch.copy()
    if c3_idx < out.shape[0] and c4_idx < out.shape[0]:
        alpha = 0.5 * severity
        avg = 0.5 * (out[c3_idx] + out[c4_idx])
        out[c3_idx] = (1 - alpha) * out[c3_idx] + alpha * avg
        out[c4_idx] = (1 - alpha) * out[c4_idx] + alpha * avg
    return out


def boost_alpha_uniform(epoch, sfreq, severity, rng):
    """Drowsy subject: uniform alpha (8-13 Hz) boost across all channels."""
    out = epoch.copy()
    for ch in range(out.shape[0]):
        alpha_band = _bandpass(out[ch], sfreq, 8, 13)
        out[ch] = out[ch] + severity * 1.5 * alpha_band
    return out


def substitute_rest_like(epoch, sfreq, severity, rng):
    """Replace a portion of the trial with matched-variance in-band noise
    to simulate 'no MI happening'. severity controls the replaced fraction."""
    n_ch, n_t = epoch.shape
    frac = 0.3 + 0.7 * severity
    n_replace = int(n_t * frac)
    start = int(rng.integers(0, max(1, n_t - n_replace)))
    noise = rng.standard_normal((n_ch, n_replace))
    noise = _bandpass(noise, sfreq, 8, 32)
    for ch in range(n_ch):
        noise[ch] = noise[ch] * _rms(epoch[ch]) / _rms(noise[ch])
    out = epoch.copy()
    out[:, start:start + n_replace] = noise
    return out


# ---------------------------------------------------------------------------
# Implausibility family
# ---------------------------------------------------------------------------

def channel_shuffle(epoch, sfreq, severity, rng):
    """Permute a subset of channels so CSP's learned spatial structure breaks."""
    n_ch = epoch.shape[0]
    n_shuffle = max(2, int(severity * n_ch))
    idx = rng.choice(n_ch, size=n_shuffle, replace=False)
    perm = rng.permutation(idx)
    out = epoch.copy()
    out[idx] = epoch[perm]
    return out


def swap_hemispheres(epoch, sfreq, severity, rng, left_idx=None, right_idx=None):
    """Mix mirrored left/right motor channels (symmetric pairs for the 22-ch
    BNCI2014_001 montage). severity controls how much each pair is swapped."""
    if left_idx is None or right_idx is None:
        left_idx = [1, 6, 7, 8, 13, 14, 18]     # FC3, C5, C3, C1, CP3, CP1, P1
        right_idx = [5, 12, 11, 10, 17, 16, 20] # FC4, C6, C4, C2, CP4, CP2, P2
    out = epoch.copy()
    alpha = 0.5 + 0.5 * severity
    for l, r in zip(left_idx, right_idx):
        if l < out.shape[0] and r < out.shape[0]:
            new_l = (1 - alpha) * out[l] + alpha * out[r]
            new_r = (1 - alpha) * out[r] + alpha * out[l]
            out[l], out[r] = new_l, new_r
    return out


# ---------------------------------------------------------------------------
# SNR degradation family
# ---------------------------------------------------------------------------

def add_inband_pink(epoch, sfreq, severity, rng):
    """1/f noise shaped in-band, scaled to a target per-channel SNR."""
    n_ch, n_t = epoch.shape
    snr_db = 20.0 - 25.0 * severity
    out = np.empty_like(epoch)
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    scale = np.where(freqs > 0, 1.0 / np.sqrt(freqs + 1e-9), 0.0)
    for ch in range(n_ch):
        white = rng.standard_normal(n_t)
        spec = np.fft.rfft(white) * scale
        pink = np.fft.irfft(spec, n=n_t)
        pink = _bandpass(pink, sfreq, 8, 32)
        signal_rms = _rms(epoch[ch])
        pink = pink * (signal_rms / (10.0 ** (snr_db / 20.0))) / _rms(pink)
        out[ch] = epoch[ch] + pink
    return out


def channel_dropout(epoch, sfreq, severity, rng):
    """Replace a few channels with matched-variance in-band noise (simulates
    a channel losing contact). A literal flatline is unrealistic post-
    bandpass — any upstream rejection pipeline would have caught it — so
    only the noise-replacement sub-mode is kept."""
    n_ch, n_t = epoch.shape
    n_drop = max(1, int(severity * 4))
    drop_idx = rng.choice(n_ch, size=min(n_drop, n_ch), replace=False)
    out = epoch.copy()
    for ch in drop_idx:
        noise = _bandpass(rng.standard_normal(n_t), sfreq, 8, 32)
        out[ch] = noise * _rms(epoch[ch]) / _rms(noise)
    return out


# ---------------------------------------------------------------------------
# Borderline family — mild / realistic failure modes
# These primitives intentionally never produce "obviously bad" trials. Their
# whole operating range sits in the near-threshold zone where the acceptance
# model has to make hard calls. Oversampling this zone is what teaches a
# sharp decision boundary.
# ---------------------------------------------------------------------------

def weaken_erd_partial(epoch, sfreq, severity, rng, motor_ch=(7, 11)):
    """Scale mu/beta power on motor channels by a factor in [0.4, 0.8] —
    simulates weak but present ERD (common in stroke / fatigued subjects).
    Distinct from attenuate_erd which can drive ERD close to zero."""
    out = epoch.copy()
    factor = 0.8 - 0.4 * severity  # sev 0 -> 0.8 (mild), sev 1 -> 0.4 (moderate)
    for ch in motor_ch:
        if ch < out.shape[0]:
            mu = _bandpass(out[ch], sfreq, 8, 13)
            beta = _bandpass(out[ch], sfreq, 13, 30)
            other = out[ch] - mu - beta
            out[ch] = other + factor * mu + factor * beta
    return out


def partial_lateralization(epoch, sfreq, severity, rng, c3_idx=7, c4_idx=11):
    """Blend C3 and C4 by a small factor in [0.10, 0.35] — simulates weakened
    but not destroyed lateralization. Distinct from destroy_lateralization
    (which goes up to 0.5 and erases the contrast entirely)."""
    out = epoch.copy()
    if c3_idx < out.shape[0] and c4_idx < out.shape[0]:
        alpha = 0.10 + 0.25 * severity
        avg = 0.5 * (out[c3_idx] + out[c4_idx])
        out[c3_idx] = (1 - alpha) * out[c3_idx] + alpha * avg
        out[c4_idx] = (1 - alpha) * out[c4_idx] + alpha * avg
    return out


def low_snr_borderline(epoch, sfreq, severity, rng):
    """Additive in-band pink noise at SNR in [8, 15] dB — the borderline
    region where decisions get hard. Distinct from add_inband_pink which
    spans 20 to -5 dB (too wide; includes both trivial and extreme cases)."""
    n_ch, n_t = epoch.shape
    snr_db = 15.0 - 7.0 * severity  # sev 0 -> 15 dB, sev 1 -> 8 dB
    out = np.empty_like(epoch)
    freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
    scale = np.where(freqs > 0, 1.0 / np.sqrt(freqs + 1e-9), 0.0)
    for ch in range(n_ch):
        white = rng.standard_normal(n_t)
        spec = np.fft.rfft(white) * scale
        pink = np.fft.irfft(spec, n=n_t)
        pink = _bandpass(pink, sfreq, 8, 32)
        signal_rms = _rms(epoch[ch])
        pink = pink * (signal_rms / (10.0 ** (snr_db / 20.0))) / _rms(pink)
        out[ch] = epoch[ch] + pink
    return out


# ---------------------------------------------------------------------------
# Engine wrapper
# ---------------------------------------------------------------------------

@dataclass
class CorruptionSpec:
    name: str
    fn: Callable
    family: str
    # "task": apply only to the imagery window. Used for ERD-related and
    #         spatial-implausibility primitives — corrupting the rest baseline
    #         too would mask the corruption from baseline-referenced features.
    # "full": apply to the whole epoch (artifacts, broadband noise, electrode
    #         failure — all of which can occur anywhere in a recording).
    domain: str = "full"


@dataclass
class CorruptionEngine:
    sfreq: float
    # Index where the rest baseline ends and the imagery window begins.
    # Default 500 = 2 s @ 250 Hz, matching the notebook's BASELINE_SAMPLES.
    baseline_samples: int = 500
    family_weights: Dict[str, float] = field(default_factory=lambda: {
        "artifact":       0.20,
        "disengagement":  0.45,
        "implausibility": 0.10,
        "snr":            0.10,
        "borderline":     0.15,
    })
    # Skewed toward mild: the decision boundary lives in the near-threshold
    # zone, so oversample it.
    severity_mix: Tuple[float, float, float] = (0.50, 0.35, 0.15)  # mild / moderate / severe
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.primitives: List[CorruptionSpec] = [
            # --- artifacts (can happen anywhere in a recording) ---
            CorruptionSpec("emg_burst",              inject_emg_burst,        "artifact",      "full"),
            CorruptionSpec("eog_residual",           inject_eog_residual,     "artifact",      "full"),
            CorruptionSpec("electrode_ring",         inject_electrode_ring,   "artifact",      "full"),
            # --- disengagement (the subject failed to engage during MI;
            #     baseline rest stays clean so ERD reference is informative) ---
            CorruptionSpec("attenuate_erd",          attenuate_erd,           "disengagement", "task"),
            CorruptionSpec("destroy_lateralization", destroy_lateralization,  "disengagement", "task"),
            CorruptionSpec("boost_alpha",            boost_alpha_uniform,     "disengagement", "full"),
            CorruptionSpec("substitute_rest",        substitute_rest_like,    "disengagement", "task"),
            # --- implausibility (spatial layout is wrong only during MI) ---
            CorruptionSpec("channel_shuffle",        channel_shuffle,         "implausibility","task"),
            CorruptionSpec("swap_hemispheres",       swap_hemispheres,        "implausibility","task"),
            # --- broadband / electrode (persists across the whole epoch) ---
            CorruptionSpec("inband_pink",            add_inband_pink,         "snr",           "full"),
            CorruptionSpec("channel_dropout",        channel_dropout,         "snr",           "full"),
            # --- borderline ERD/lateralization weakening ---
            CorruptionSpec("weaken_erd_partial",     weaken_erd_partial,      "borderline",    "task"),
            CorruptionSpec("partial_lateralization", partial_lateralization,  "borderline",    "task"),
            CorruptionSpec("low_snr_borderline",     low_snr_borderline,      "borderline",    "full"),
        ]
        self._by_family: Dict[str, List[CorruptionSpec]] = {}
        for p in self.primitives:
            self._by_family.setdefault(p.family, []).append(p)

    def _sample_severity(self) -> float:
        bucket = int(self.rng.choice(3, p=list(self.severity_mix)))
        lo, hi = [(0.0, 0.33), (0.33, 0.66), (0.66, 1.0)][bucket]
        return float(self.rng.uniform(lo, hi))

    def _sample_primitive(self) -> CorruptionSpec:
        families = list(self.family_weights.keys())
        probs = np.array([self.family_weights[f] for f in families], dtype=float)
        probs = probs / probs.sum()
        fam = str(self.rng.choice(families, p=probs))
        return self._by_family[fam][int(self.rng.integers(0, len(self._by_family[fam])))]

    def corrupt_one(self, epoch: np.ndarray):
        prim = self._sample_primitive()
        sev = self._sample_severity()

        if prim.domain == "task":
            # Apply the primitive to the imagery slice only; reassemble the
            # full epoch so the rest baseline is preserved unchanged.
            full = epoch.copy()
            task_in = full[..., self.baseline_samples:].copy()
            task_out = prim.fn(task_in, self.sfreq, sev, self.rng)
            full[..., self.baseline_samples:] = task_out
            corrupted = full
        else:
            corrupted = prim.fn(epoch.copy(), self.sfreq, sev, self.rng)

        return corrupted, {"corruption": prim.name, "family": prim.family,
                           "severity": sev, "domain": prim.domain}

    def generate_dataset(self, clean_epochs: np.ndarray, n_per_epoch: int = 5):
        X_bad = np.empty((len(clean_epochs) * n_per_epoch, *clean_epochs.shape[1:]), dtype=clean_epochs.dtype)
        meta: List[dict] = []
        k = 0
        for i, epoch in enumerate(clean_epochs):
            for _ in range(n_per_epoch):
                corrupted, m = self.corrupt_one(epoch)
                X_bad[k] = corrupted
                m["source_epoch_idx"] = i
                meta.append(m)
                k += 1
        return X_bad, meta
