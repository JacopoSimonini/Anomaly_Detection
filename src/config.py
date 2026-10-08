"""
Configuration for the anomaly detection pipeline.

All hyperparameters and paths are centralized here for easy experimentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessConfig:
    """Audio preprocessing parameters."""

    sr: int = 16000  # Sample rate (Hz)

    # Windowing
    window_sec: float = 2.0  # Window duration (2s for broader temporal context)
    hop_sec: float = 1.0     # Hop size in seconds (50% overlap)

    # Mel spectrogram
    n_mels: int = 128        # Number of mel bands (standard)
    n_fft: int = 1024        # FFT window size ~64ms analysis window
    hop_length: int = 512    # STFT hop length (samples)
    fmin: float = 0.0        # Minimum frequency for mel filterbank
    fmax: Optional[float] = None  # Maximum frequency (None = sr/2 = 8kHz)

    # Normalization
    normalize: bool = True   # Apply normalization to [0, 1]
    log_scale: bool = True   # Convert to log-mel spectrogram

    # Global normalization: uses fixed dB range instead of per-spectrogram min/max.
    # db_min/db_max should be set from data percentiles (see compute_db_percentiles).
    global_norm: bool = True
    db_min: float = -46.1    # Noise floor (1st percentile from training data)
    db_max: float = -10.0    # Signal ceiling (99th percentile from training data)

    # Fixed reference for dB conversion (preserves absolute intensity across spectrograms).
    # If True, uses ref=1.0; if False, uses ref=np.max (per-spectrogram, loses intensity info).
    use_fixed_ref: bool = True

    @property
    def window_samples(self) -> int:
        """Window size in samples."""
        return int(self.window_sec * self.sr)

    @property
    def hop_samples(self) -> int:
        """Hop size in samples."""
        return int(self.hop_sec * self.sr)