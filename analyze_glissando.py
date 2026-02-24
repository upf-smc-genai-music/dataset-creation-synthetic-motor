"""
analyze_glissandi_librosa.py  –  pYIN f0 analysis for violin glissando recordings
=================================================================================

Usage
-----
    # Single file:
    python analyze_glissandi_librosa.py path/to/recording.wav

    # All WAVs in a folder:
    python analyze_glissandi_librosa.py path/to/folder/

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

Algorithm: librosa.pyin  (Mauch & Dixon, ICASSP 2014)
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TARGET_SR        = 44100
FRAME_RATE       = 75
FMIN             = 130.0
FMAX             = 1400.0
VOICED_THRESHOLD = 0.40

WIN_MS           = 64.0


# ─────────────────────────────────────────────────────────────────────────────
# Mel helpers
# ─────────────────────────────────────────────────────────────────────────────

def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path, target_sr=TARGET_SR):
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return sr, audio.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# pYIN analysis (librosa)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_pyin(audio, sr):
    """
    Run librosa.pyin on mono float64 audio.
    Returns a DataFrame at FRAME_RATE fps.
    """

    hop_length = int(round(sr / FRAME_RATE))
    win_length = int(sr * WIN_MS / 1000.0)

    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        frame_length=win_length,
        hop_length=hop_length,
        center=True,
    )

    # Convert to numpy arrays
    f0 = np.asarray(f0)
    voiced_prob = np.asarray(voiced_prob)

    # Apply voiced probability threshold
    voiced = voiced_prob >= VOICED_THRESHOLD
    f0[~voiced] = np.nan

    # Time axis
    times = librosa.frames_to_time(
        np.arange(len(f0)),
        sr=sr,
        hop_length=hop_length,
    )

    # Mel conversion
    f0_mel = np.where(voiced, hz_to_mel(f0), np.nan)

    return pd.DataFrame({
        "time_sec":      times,
        "frequency_hz":  f0,
        "frequency_mel": f0_mel,
        "confidence":    voiced_prob,
        "voiced":        voiced,
    })


# ─────────────────────────────────────────────────────────────────────────────
# File handling
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(wav_path, verbose=True):
    if verbose:
        print(f"  → {wav_path}")

    sr, audio = load_audio(wav_path)
    df = analyze_pyin(audio, sr)

    csv_path = os.path.splitext(wav_path)[0] + ".csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")

    if verbose:
        v = df["voiced"]
        hz = df.loc[v, "frequency_hz"]
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
        print("  python analyze_glissandi_librosa.py recording.wav")
        print("  python analyze_glissandi_librosa.py folder/with/wavs/")
        sys.exit(0)

    for arg in sys.argv[1:]:
        process_path(arg)