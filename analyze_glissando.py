"""
analyze_glissandi.py  –  pYIN f0 analysis for violin glissando recordings
=========================================================================

Usage
-----
    # Single file:
    python analyze_glissandi.py path/to/recording.wav

    # All WAVs in a folder:
    python analyze_glissandi.py path/to/folder/

Output
------
For every  <name>.wav  →  <name>.csv  in the same directory.

CSV columns
-----------
    time_sec        centre time of each analysis frame (seconds)
    frequency_hz    estimated f0 in Hz  (NaN if unvoiced / below threshold)
    frequency_mel   same converted to mel scale  (NaN if unvoiced)
    confidence      voicing probability [0, 1]
    voiced          bool – True when confidence ≥ VOICED_THRESHOLD

Algorithm: pYIN  (Mauch & Dixon, ICASSP 2014)
----------------------------------------------
  1. Frame audio with a Hann window
  2. Compute the CMNDF (cumulative mean normalised difference function)
  3. Parabolic interpolation → sub-sample trough positions (f0 candidates)
  4. Threshold sweep weighted by a Beta prior → probability distribution over
     candidates and a voiced/unvoiced split
  5. HMM Viterbi decoding enforces pitch continuity and voiced/unvoiced
     transitions – the key improvement over plain YIN for glissandi

Dependencies: numpy, scipy, pandas  (all standard – no network required)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR           = 44100
FRAME_RATE          = 70          # output frames per second
FMIN                = 130.0       # Hz  (below violin G3)
FMAX                = 1400.0      # Hz  (above violin E7)
VOICED_THRESHOLD    = 0.40        # min voicing probability to emit a pitch

WIN_MS              = 64.0        # analysis window (ms)
N_THRESHOLDS        = 100
THRESHOLD_MIN       = 0.01
THRESHOLD_MAX       = 0.20
BETA_A              = 2.0         # Beta(a, b) prior on thresholds
BETA_B              = 18.0

# HMM parameters
TRANSITION_SIGMA    = 1.0         # semitones – Gaussian width for pitch jumps
                                  # (unnormalized, so only the shape matters)
VOICED_VOICED_P     = 0.99
UNVOICED_UNVOICED_P = 0.90
N_BINS_PER_SEMITONE = 5           # pitch grid resolution


# ─────────────────────────────────────────────────────────────────────────────
# Mel helpers
# ─────────────────────────────────────────────────────────────────────────────

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)

def hz_to_semitone(f):
    """Continuous semitone number (A4 = 69)."""
    return 12.0 * np.log2(np.asarray(f, dtype=float) / 440.0) + 69.0


# ─────────────────────────────────────────────────────────────────────────────
# Audio I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path, target_sr=TARGET_SR):
    sr, data = wavfile.read(path)

    # Mono + float64 [-1, 1]
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)

    if int(sr) != int(target_sr):
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g)
        sr = target_sr

    return int(sr), data


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def next_pow2(n):
    return 1 << (int(n) - 1).bit_length()

def beta_pdf(x, a, b):
    x = np.asarray(x, dtype=float)
    return x ** (a - 1) * (1.0 - x) ** (b - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1-3: CMNDF + local trough detection
# ─────────────────────────────────────────────────────────────────────────────

def compute_cmndf(frame, tau_min, tau_max):
    """
    Compute the Cumulative Mean Normalised Difference Function and find
    all local troughs in [tau_min, tau_max) with parabolic interpolation.

    Returns
    -------
    cmndf   : np.ndarray shape (tau_max,)
    troughs : list of (tau_float, cmndf_value) sorted by tau
    """
    N       = len(frame)
    tau_max = min(tau_max, N // 2)

    # Difference function d(τ) = Σ (x_t − x_{t+τ})²
    diff    = np.empty(tau_max, dtype=float)
    diff[0] = 0.0
    for tau in range(1, tau_max):
        r        = frame[: N - tau] - frame[tau :]
        diff[tau] = float(np.dot(r, r))

    # CMNDF: normalise d(τ) by its running cumulative mean
    cmndf   = np.ones(tau_max, dtype=float)
    running = 0.0
    for tau in range(1, tau_max):
        running   += diff[tau]
        cmndf[tau] = diff[tau] * tau / running if running > 1e-12 else 1.0

    # Local trough detection with parabolic interpolation
    troughs = []
    for tau in range(max(tau_min, 2), tau_max - 1):
        a, b, c = cmndf[tau - 1], cmndf[tau], cmndf[tau + 1]
        if b < a and b < c:
            denom  = a - 2.0 * b + c
            offset = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            offset = float(np.clip(offset, -0.5, 0.5))
            tau_f  = float(tau) + offset
            val    = float(b) - 0.25 * (a - c) * offset
            troughs.append((tau_f, val))

    return cmndf, troughs


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: pYIN threshold sweep → per-frame candidate distribution
# ─────────────────────────────────────────────────────────────────────────────

def pyin_candidates(troughs, thresholds, threshold_priors):
    """
    Marginalise over YIN thresholds (weighted by a Beta prior) to obtain:
      - a voiced probability p_voiced
      - a conditional distribution over candidate tau values (given voiced)

    Returns
    -------
    candidates : list of (tau_float, probability)  summing to 1.0
    p_voiced   : float
    """
    voiced_mass = {}
    p_voiced    = 0.0
    p_unvoiced  = 0.0

    for thresh, p_t in zip(thresholds, threshold_priors):
        picked = None
        for tau, val in troughs:          # troughs are sorted by ascending tau
            if val < thresh:
                picked = tau
                break

        p_t = float(p_t)
        if picked is not None:
            key                 = round(float(picked), 3)
            voiced_mass[key]    = voiced_mass.get(key, 0.0) + p_t
            p_voiced           += p_t
        else:
            p_unvoiced += p_t

    total = p_voiced + p_unvoiced
    if total < 1e-12:
        return [], 0.0

    p_voiced /= total

    candidates = list(voiced_mass.items())
    s = sum(p for _, p in candidates)
    if s > 0:
        candidates = [(tau, p / s) for tau, p in candidates]

    return candidates, p_voiced


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: HMM Viterbi decoding
# ─────────────────────────────────────────────────────────────────────────────

def build_pitch_grid(sr, tau_min, tau_max):
    """Uniform semitone grid covering [FMIN, FMAX]."""
    st_lo     = float(hz_to_semitone(sr / tau_max))
    st_hi     = float(hz_to_semitone(sr / tau_min))
    step      = 1.0 / N_BINS_PER_SEMITONE
    semitones = np.arange(st_lo, st_hi + step, step)
    return semitones


def viterbi_decode(frame_candidates, frame_voiced_probs, semitones, sr):
    """
    HMM Viterbi over states [unvoiced, pitch_0 … pitch_{N-1}].

    Pitch-to-pitch transitions use an **unnormalized** log-Gaussian kernel.
    This is correct for Viterbi (only relative log-probabilities matter)
    and avoids the numerical decay that normalized row-stochastic matrices
    cause when the grid is much finer than the transition width.

    Returns
    -------
    f0_hz  : np.ndarray (T,)  – NaN for unvoiced frames
    voiced : np.ndarray bool  (T,)
    conf   : np.ndarray float (T,)  – voicing probability per frame
    """
    N_p = len(semitones)
    T   = len(frame_candidates)
    NEG = -1e30

    # ── Unnormalized log-Gaussian transition kernel  (N_p × N_p) ──────────
    # log_trans[j, i] = −0.5 · ((st_j − st_i) / σ)²
    # For Viterbi only differences matter; the constant −0.5·0²/σ² = 0 is
    # the maximum (self-transition), so every cross-state is penalised.
    st_diff   = semitones[:, None] - semitones[None, :]    # (N_p, N_p)
    log_trans = -0.5 * (st_diff / TRANSITION_SIGMA) ** 2   # max = 0 at diagonal

    log_vv  = np.log(VOICED_VOICED_P)
    log_vu  = np.log(1.0 - VOICED_VOICED_P)
    log_uu  = np.log(UNVOICED_UNVOICED_P)
    log_uv  = np.log(1.0 - UNVOICED_UNVOICED_P) - np.log(float(N_p))

    # ── Viterbi tables ────────────────────────────────────────────────────
    N_states = 1 + N_p
    V        = np.full((T, N_states), NEG)
    B        = np.zeros((T, N_states), dtype=np.int32)

    # Initial state: equal probability between voiced and unvoiced
    V[0, 0]  = np.log(0.5)
    V[0, 1:] = np.log(0.5 / N_p)

    for t in range(T):
        cands = frame_candidates[t]
        p_v   = float(frame_voiced_probs[t])

        # ── Emission log-probs ────────────────────────────────────────────
        emit    = np.full(N_states, NEG)
        emit[0] = np.log(max(1.0 - p_v, 1e-12))

        for tau, prob in cands:
            if tau <= 0:
                continue
            st  = float(hz_to_semitone(sr / tau))
            idx = int(np.argmin(np.abs(semitones - st)))
            lp  = np.log(max(p_v * float(prob), 1e-30))
            if lp > emit[1 + idx]:
                emit[1 + idx] = lp

        if t == 0:
            V[0] += emit
            continue

        prev = V[t - 1]

        # ── Unvoiced state ────────────────────────────────────────────────
        s_uu = prev[0] + log_uu
        s_pu = float(np.max(prev[1:])) + log_vu
        if s_uu >= s_pu:
            V[t, 0] = s_uu + emit[0]
            B[t, 0] = 0
        else:
            V[t, 0] = s_pu + emit[0]
            B[t, 0] = int(np.argmax(prev[1:])) + 1

        # ── Pitch states (vectorised over destination pitch i) ────────────
        from_u = prev[0] + log_uv          # scalar – uniform over all pitches

        # from_p_mat[j, i] = prev[j+1] + log_vv + log_trans[j, i]
        from_p_mat      = prev[1:, None] + log_vv + log_trans  # (N_p, N_p)
        best_from_p     = from_p_mat.max(axis=0)               # (N_p,)
        best_from_p_idx = from_p_mat.argmax(axis=0) + 1        # source idx

        voiced_wins = best_from_p >= from_u
        V[t, 1:]    = np.where(voiced_wins, best_from_p, from_u) + emit[1:]
        B[t, 1:]    = np.where(voiced_wins, best_from_p_idx, 0)

    # ── Backtrack ─────────────────────────────────────────────────────────
    states       = np.zeros(T, dtype=np.int32)
    states[T - 1] = int(np.argmax(V[T - 1]))
    for t in range(T - 2, -1, -1):
        states[t] = B[t + 1, states[t + 1]]

    # ── Convert states → Hz ───────────────────────────────────────────────
    f0_hz  = np.full(T, np.nan)
    voiced = np.zeros(T, dtype=bool)
    conf   = np.asarray(frame_voiced_probs, dtype=float)

    for t in range(T):
        s = states[t]
        if s > 0:
            f0_hz[t]  = 440.0 * 2.0 ** ((semitones[s - 1] - 69.0) / 12.0)
            voiced[t] = True

    # Gate by threshold
    below          = conf < VOICED_THRESHOLD
    f0_hz[below]   = np.nan
    voiced[below]  = False

    return f0_hz, voiced, conf


# ─────────────────────────────────────────────────────────────────────────────
# Top-level pYIN
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pyin(audio, sr):
    """Run pYIN on a mono float64 array. Returns a DataFrame at FRAME_RATE fps."""
    hop      = int(round(sr / FRAME_RATE))
    win_size = next_pow2(int(sr * WIN_MS / 1000.0))
    tau_min  = max(2, int(sr / FMAX))
    tau_max  = min(win_size // 2, int(sr / FMIN) + 1)

    window           = np.hanning(win_size)
    thresholds       = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, N_THRESHOLDS)
    threshold_priors = beta_pdf(thresholds, BETA_A, BETA_B)
    threshold_priors /= threshold_priors.sum()

    # Reflect-pad so edge frames are centred on audio boundaries
    pad       = win_size // 2
    audio_pad = np.pad(audio, (pad, pad + win_size), mode='reflect')
    n_frames  = int(np.ceil(len(audio) / hop))

    all_cands  = []
    all_pvoice = []

    for i in range(n_frames):
        start = i * hop
        frame = audio_pad[start : start + win_size].copy()
        if len(frame) < win_size:
            frame = np.pad(frame, (0, win_size - len(frame)))
        frame *= window

        _, troughs = compute_cmndf(frame, tau_min, tau_max)
        cands, p_v = pyin_candidates(troughs, thresholds, threshold_priors)
        all_cands.append(cands)
        all_pvoice.append(p_v)

    semitones         = build_pitch_grid(sr, tau_min, tau_max)
    f0_hz, voiced, conf = viterbi_decode(all_cands, all_pvoice, semitones, sr)

    times  = (np.arange(n_frames) * hop + hop / 2.0) / sr
    f0_mel = np.where(voiced, hz_to_mel(np.where(voiced, f0_hz, 1.0)), np.nan)

    return pd.DataFrame({
        "time_sec":      times,
        "frequency_hz":  f0_hz,
        "frequency_mel": f0_mel,
        "confidence":    conf,
        "voiced":        voiced,
    })


# ─────────────────────────────────────────────────────────────────────────────
# File handling
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(wav_path, verbose=True):
    if verbose:
        print(f"  → {wav_path}")

    sr, audio = load_audio(wav_path)
    df        = analyze_pyin(audio, sr)

    csv_path = os.path.splitext(wav_path)[0] + ".csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    if verbose:
        v   = df["voiced"]
        hz  = df.loc[v, "frequency_hz"]
        pct = v.mean() * 100
        rng = (f"{hz.min():.1f}–{hz.max():.1f} Hz"
               if v.any() else "no voiced frames")
        print(f"     {len(df)} frames | {pct:.1f}% voiced | f0 {rng}")
        print(f"     saved → {csv_path}")

    return csv_path


def process_path(path):
    if os.path.isfile(path):
        if path.lower().endswith(".wav"):
            analyze_file(path)
        else:
            print(f"Skipping (not a .wav): {path}")
    elif os.path.isdir(path):
        wavs = sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".wav")
        )
        if not wavs:
            print(f"No .wav files found in '{path}'")
            return
        print(f"Found {len(wavs)} WAV file(s) in '{path}'")
        for w in wavs:
            analyze_file(w)
        print("Done.")
    else:
        print(f"Path not found: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage:")
        print("  python analyze_glissandi.py recording.wav")
        print("  python analyze_glissandi.py folder/with/wavs/")
        sys.exit(0)

    for arg in sys.argv[1:]:
        process_path(arg)