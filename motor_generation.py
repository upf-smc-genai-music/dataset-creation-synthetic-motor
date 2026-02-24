import os
import numpy as np
import pandas as pd
from scipy.io import wavfile

# ==========================
# Configuration
# ==========================

sr = 44100
rpm_min = 4000
rpm_max = 15000

total_duration_sec = 900      # 15 minutes
segment_duration_sec = 4     # 4 sec per file
n_harmonics = 20

output_folder = "raw"
os.makedirs(output_folder, exist_ok=True)

# Number of files
n_files = total_duration_sec // segment_duration_sec

# Linearly spaced RPM values
rpm_values = np.linspace(rpm_min, rpm_max, n_files)

# ==========================
# Motor Synthesis Function
# ==========================

def generate_motor_sound_from_rpm(rpm,
                                  duration=10.0,
                                  sr=44100,
                                  n_harmonics=20):
    """
    Generate synthetic motor sound from actual RPM value.
    """

    combustion_events_per_rev = 2  # 4-cylinder 4-stroke approximation
    f0 = (rpm / 60.0) * combustion_events_per_rev

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Harmonic engine tone
    signal = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        amplitude = 1.0 / k
        signal += amplitude * np.sin(2 * np.pi * k * f0 * t)

    # RPM-dependent noise
    alpha = (rpm - rpm_min) / (rpm_max - rpm_min)
    noise = np.random.randn(len(t))
    noise_level = 0.05 + 0.25 * alpha
    signal += noise_level * noise

    # Brightness scaling
    brightness = 0.5 + alpha
    signal = np.tanh(brightness * signal)

    # Normalize
    signal /= np.max(np.abs(signal))

    # Convert to int16
    signal_int16 = np.int16(signal * 32767)

    return signal_int16


# ==========================
# Dataset Generation
# ==========================

# ==========================
# Dataset Generation
# ==========================

FRAME_RATE = 75  # frames per second

for i, rpm in enumerate(rpm_values):
    base_name = f"motor_{i:03d}"
    wav_path = os.path.join(output_folder, base_name + ".wav")
    csv_path = os.path.join(output_folder, base_name + ".csv")

    audio = generate_motor_sound_from_rpm(
        rpm=rpm,
        duration=segment_duration_sec,
        sr=sr,
        n_harmonics=n_harmonics
    )

    # Save WAV
    wavfile.write(wav_path, sr, audio)

    # Number of parameter frames
    n_frames = int(FRAME_RATE * segment_duration_sec)

    # Create constant RPM column at 75 fps resolution
    df = pd.DataFrame({
        "rpm": np.full(n_frames, rpm)
    })

    df.to_csv(csv_path, index=False, float_format="%.4f")

print(f"Generated {n_files} files in '{output_folder}' folder.")
