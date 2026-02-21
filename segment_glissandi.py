"""
segment_glissandi.py  –  Split a long violin recording into individual glissandi
=================================================================================

Given a WAV file and its pYIN CSV (produced by analyze_glissandi.py), this
script finds the boundaries between glissandi by looking for long unvoiced
gaps, then extracts each segment as a separate WAV + CSV pair.

Usage
-----
    python segment_glissandi.py recording.wav
    python segment_glissandi.py recording.wav --min_gap_sec 0.4
    python segment_glissandi.py recording.wav --out_dir my_folder --min_gap_sec 0.4

The CSV must have the same name as the WAV (produced by analyze_glissandi.py).

Output
------
    raw_violin/
        glissando_1.wav  +  glissando_1.csv
        glissando_2.wav  +  glissando_2.csv
        ...

Arguments
---------
    wav_path        Path to the recorded WAV file
    --min_gap_sec   Minimum unvoiced gap duration (seconds) that counts as a
                    boundary between glissandi. Default: 0.40 s
                    Shorter dropouts inside a glissando are ignored.
    --out_dir       Output folder. Default: raw_violin
    --min_dur_sec   Minimum duration of a kept segment (seconds). Default: 0.5
                    Segments shorter than this are discarded (e.g. noise bursts).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_GAP_SEC  = 0.40   # gaps longer than this → glissando boundary
DEFAULT_MIN_DUR_SEC  = 0.50   # discard segments shorter than this
DEFAULT_OUT_DIR      = "raw_violin"


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation logic
# ─────────────────────────────────────────────────────────────────────────────

def find_segments(df, min_gap_frames):
    """
    Walk through the voiced column and collect (start_frame, end_frame) pairs
    for each continuous voiced region, merging across short internal dropouts
    (< min_gap_frames) and cutting only at long silences (>= min_gap_frames).

    Returns list of (start_frame, end_frame) inclusive, in frame units.
    """
    voiced = df["voiced"].values.astype(bool)
    n = len(voiced)

    # Label every frame: voiced=True, unvoiced=False
    # Find runs of unvoiced frames and their lengths
    segments = []
    seg_start = None

    i = 0
    while i < n:
        if voiced[i]:
            # Start of a voiced region
            if seg_start is None:
                seg_start = i
            i += 1
        else:
            # Unvoiced run — measure its length
            j = i
            while j < n and not voiced[j]:
                j += 1
            gap_len = j - i  # frames

            if seg_start is None:
                # Haven't started a segment yet, skip leading silence
                i = j
                continue

            if gap_len >= min_gap_frames:
                # Long gap → end current segment here, start fresh after gap
                segments.append((seg_start, i - 1))
                seg_start = None
                i = j
            else:
                # Short dropout — treat as part of the ongoing segment, skip over it
                i = j

    # Flush last segment
    if seg_start is not None:
        # Find last voiced frame
        last_voiced = np.where(voiced[seg_start:])[0]
        if len(last_voiced) > 0:
            segments.append((seg_start, seg_start + last_voiced[-1]))

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Audio slicing
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path):
    sr, data = wavfile.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1).astype(data.dtype)
    return sr, data


def frames_to_samples(frame_idx, sr, frame_rate):
    """Convert frame index to sample index (frame centre)."""
    hop = sr / frame_rate
    return int(round(frame_idx * hop + hop / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def segment(wav_path,
            min_gap_sec=DEFAULT_MIN_GAP_SEC,
            min_dur_sec=DEFAULT_MIN_DUR_SEC,
            out_dir=DEFAULT_OUT_DIR):

    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.splitext(wav_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run analyze_glissandi.py on the WAV first.")

    df = pd.read_csv(csv_path)
    # Ensure voiced is boolean (read_csv may load it as string)
    df["voiced"] = df["voiced"].astype(str).str.strip().str.lower() == "true"

    # Infer frame rate from time column
    if len(df) > 1:
        frame_rate = round(1.0 / (df["time_sec"].iloc[1] - df["time_sec"].iloc[0]))
    else:
        frame_rate = 70
    print(f"Detected frame rate: {frame_rate} fps")

    min_gap_frames = int(round(min_gap_sec * frame_rate))
    min_dur_frames = int(round(min_dur_sec * frame_rate))
    print(f"Boundary threshold: gaps ≥ {min_gap_frames} frames "
          f"({min_gap_sec*1000:.0f} ms)")

    # ── Find segments ─────────────────────────────────────────────────────────
    raw_segments = find_segments(df, min_gap_frames)
    print(f"Found {len(raw_segments)} candidate segments before duration filter")

    # Filter by minimum duration
    segments = [(s, e) for s, e in raw_segments if (e - s + 1) >= min_dur_frames]
    print(f"Kept {len(segments)} segments (≥ {min_dur_sec:.1f}s each)")

    if not segments:
        print("No segments found. Try lowering --min_gap_sec.")
        return

    # ── Load WAV ──────────────────────────────────────────────────────────────
    sr, audio = load_wav(wav_path)
    total_samples = len(audio)

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    for idx, (f_start, f_end) in enumerate(segments, start=1):
        # Convert frames → samples with a small margin (half a hop on each side)
        s_start = max(0, frames_to_samples(f_start, sr, frame_rate)
                      - int(sr / frame_rate // 2))
        s_end   = min(total_samples,
                      frames_to_samples(f_end, sr, frame_rate)
                      + int(sr / frame_rate // 2))

        audio_seg = audio[s_start:s_end]

        # Slice the CSV rows for this segment
        df_seg = df.iloc[f_start : f_end + 1].copy().reset_index(drop=True)
        # Re-zero time_sec so it starts at 0
        df_seg["time_sec"] = df_seg["time_sec"] - df_seg["time_sec"].iloc[0]

        # Compute stats while voiced column is still present
        t_start    = df["time_sec"].iloc[f_start]
        t_end      = df["time_sec"].iloc[f_end]
        dur        = t_end - t_start
        voiced_pct = df_seg["voiced"].mean() * 100
        hz_voiced  = df_seg.loc[df_seg["voiced"], "frequency_hz"]
        hz_range   = (f"{hz_voiced.min():.1f}–{hz_voiced.max():.1f} Hz"
                      if not hz_voiced.empty else "—")

        # Fill spurious unvoiced frames with the previous known frequency
        # (forward-fill, then backward-fill to handle any leading NaNs)
        df_seg["frequency_hz"] = (df_seg["frequency_hz"]
                                  .ffill()
                                  .bfill())
        # Keep only the two columns needed
        df_seg = df_seg[["time_sec", "frequency_hz"]]

        base     = f"glissando_{idx}"
        out_wav  = os.path.join(out_dir, base + ".wav")
        out_csv  = os.path.join(out_dir, base + ".csv")

        wavfile.write(out_wav, sr, audio_seg)
        df_seg.to_csv(out_csv, index=False, float_format="%.4f")

        print(f"  [{idx:>2}] t={t_start:.2f}–{t_end:.2f}s  "
              f"dur={dur:.2f}s  voiced={voiced_pct:.0f}%  "
              f"f0={hz_range}  → {base}.*")

    print(f"\nDone. {len(segments)} segments saved to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segment a long violin recording into individual glissandi.")
    parser.add_argument("wav_path",
                        help="Path to the WAV file (CSV must be alongside it)")
    parser.add_argument("--min_gap_sec", type=float, default=DEFAULT_MIN_GAP_SEC,
                        help=f"Min silence gap to split on (default: {DEFAULT_MIN_GAP_SEC}s)")
    parser.add_argument("--min_dur_sec", type=float, default=DEFAULT_MIN_DUR_SEC,
                        help=f"Min segment duration to keep (default: {DEFAULT_MIN_DUR_SEC}s)")
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR,
                        help=f"Output folder (default: {DEFAULT_OUT_DIR})")
    args = parser.parse_args()

    if not os.path.exists(args.wav_path):
        print(f"File not found: {args.wav_path}")
        sys.exit(1)

    segment(args.wav_path,
            min_gap_sec=args.min_gap_sec,
            min_dur_sec=args.min_dur_sec,
            out_dir=args.out_dir)


if __name__ == "__main__":
    main()