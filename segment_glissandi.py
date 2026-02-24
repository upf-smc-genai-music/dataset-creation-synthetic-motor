"""
segment_glissandi.py  –  Split a long violin recording into individual glissandi
=================================================================================

Same as before, but output CSV files contain ONLY:

    frequency_hz

No time column is written.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy.io import wavfile


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_GAP_SEC  = 0.40
DEFAULT_MIN_DUR_SEC  = 0.50
DEFAULT_OUT_DIR      = "raw_violin"


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation logic
# ─────────────────────────────────────────────────────────────────────────────

def find_segments(df, min_gap_frames):
    voiced = df["voiced"].values.astype(bool)
    n = len(voiced)

    segments = []
    seg_start = None

    i = 0
    while i < n:
        if voiced[i]:
            if seg_start is None:
                seg_start = i
            i += 1
        else:
            j = i
            while j < n and not voiced[j]:
                j += 1
            gap_len = j - i

            if seg_start is None:
                i = j
                continue

            if gap_len >= min_gap_frames:
                segments.append((seg_start, i - 1))
                seg_start = None
                i = j
            else:
                i = j

    if seg_start is not None:
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
    hop = sr / frame_rate
    return int(round(frame_idx * hop + hop / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def segment(wav_path,
            min_gap_sec=DEFAULT_MIN_GAP_SEC,
            min_dur_sec=DEFAULT_MIN_DUR_SEC,
            out_dir=DEFAULT_OUT_DIR):

    csv_path = os.path.splitext(wav_path)[0] + ".csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run analyze_glissandi.py on the WAV first.")

    df = pd.read_csv(csv_path)
    df["voiced"] = df["voiced"].astype(str).str.strip().str.lower() == "true"

    # Infer frame rate from CSV length (no time column needed)
    total_frames = len(df)
    sr, audio = load_wav(wav_path)
    frame_rate = round(total_frames / (len(audio) / sr))

    print(f"Detected frame rate: {frame_rate} fps")

    min_gap_frames = int(round(min_gap_sec * frame_rate))
    min_dur_frames = int(round(min_dur_sec * frame_rate))

    raw_segments = find_segments(df, min_gap_frames)
    print(f"Found {len(raw_segments)} candidate segments")

    segments = [(s, e) for s, e in raw_segments if (e - s + 1) >= min_dur_frames]
    print(f"Kept {len(segments)} segments (≥ {min_dur_sec:.1f}s each)")

    if not segments:
        print("No segments found. Try lowering --min_gap_sec.")
        return

    total_samples = len(audio)
    os.makedirs(out_dir, exist_ok=True)

    for idx, (f_start, f_end) in enumerate(segments, start=1):

        s_start = max(0, frames_to_samples(f_start, sr, frame_rate)
                      - int(sr / frame_rate // 2))
        s_end   = min(total_samples,
                      frames_to_samples(f_end, sr, frame_rate)
                      + int(sr / frame_rate // 2))

        audio_seg = audio[s_start:s_end]

        df_seg = df.iloc[f_start : f_end + 1].copy().reset_index(drop=True)

        # Fill spurious unvoiced frames
        df_seg["frequency_hz"] = df_seg["frequency_hz"].ffill().bfill()

        # Keep ONLY frequency column
        df_seg = df_seg[["frequency_hz"]]

        base     = f"glissando_{idx}"
        out_wav  = os.path.join(out_dir, base + ".wav")
        out_csv  = os.path.join(out_dir, base + ".csv")

        wavfile.write(out_wav, sr, audio_seg)
        df_seg.to_csv(out_csv, index=False, float_format="%.4f")

        print(f"  [{idx:>2}] frames={f_end - f_start + 1}  → {base}.*")

    print(f"\nDone. {len(segments)} segments saved to '{out_dir}/'")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segment a long violin recording into individual glissandi.")
    parser.add_argument("wav_path",
                        help="Path to the WAV file (CSV must be alongside it)")
    parser.add_argument("--min_gap_sec", type=float, default=DEFAULT_MIN_GAP_SEC)
    parser.add_argument("--min_dur_sec", type=float, default=DEFAULT_MIN_DUR_SEC)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
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