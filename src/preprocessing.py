import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import librosa
import json
from tqdm import tqdm

from .config import PreprocessConfig

logger = logging.getLogger(__name__)


def compute_mel_spectrogram(segment: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """
    Convert audio segment to log-Mel spectrogram.

    Args:
        segment: 1D audio signal
        config: Preprocessing configuration

    Returns:
        2D spectrogram array of shape (n_mels, time_frames)
    """
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=config.sr,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax
    )

    # Convert to log scale
    if config.log_scale:
        # use_fixed_ref=True: Use ref=1.0 to preserve absolute intensity across spectrograms
        # use_fixed_ref=False: Use ref=np.max (per-spectrogram, loses intensity info)
        if config.use_fixed_ref:
            mel_spec = librosa.power_to_db(mel_spec, ref=1.0)
        else:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1] range
    if config.normalize:
        # --- Global normalization (fixed dB range) ---
        # db_min/db_max should be set based on data percentiles (see compute_db_percentiles)
        # Preserves relative intensity information across spectrograms
        if config.global_norm:
            mel_spec = np.clip(mel_spec, config.db_min, config.db_max)
            mel_spec = (mel_spec - config.db_min) / (config.db_max - config.db_min + 1e-8)
        # --- Per-spectrogram normalization (original) ---
        # Each spectrogram normalized independently to [0,1]
        else:
            mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)

    # Pad time axis to next power of 2 for clean conv encoder/decoder dimension halving
    n_frames = mel_spec.shape[1]
    next_pow2 = 1 << (n_frames - 1).bit_length()
    if n_frames < next_pow2:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, next_pow2 - n_frames)), mode='constant')

    return mel_spec


def process_file(file_path: Path, config: PreprocessConfig) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Process a single audio file into mel spectrograms.

    Args:
        file_path: Path to audio file
        config: Preprocessing configuration

    Returns:
        Tuple of:
            - List of spectrograms (each of shape (n_mels, time_frames))
            - List of metadata dicts with source info
    """
    file_path = Path(file_path)

    # Load audio
    waveform, _ = librosa.load(file_path, sr=config.sr, mono=True)

    # Extract windows
    windows = []
    start = 0
    window_samples=config.window_samples
    hop_samples=config.hop_samples

    while start + window_samples <= len(waveform):
        window = waveform[start:start + window_samples]
        windows.append(window)
        start += hop_samples

    # Convert each window to spectrogram
    spectrograms = []
    metadata = []

    for idx, window in enumerate(windows):
        spec = compute_mel_spectrogram(window, config)
        spectrograms.append(spec)
        source_dir = file_path.parent.name
        metadata.append({
            "source_file": file_path.name,
            "source_dir": source_dir,
            "source_id": f"{source_dir}/{file_path.name}",
            "window_idx": idx,
            "start_sec": idx * config.hop_sec,
            "end_sec": idx * config.hop_sec + config.window_sec
        })

    return spectrograms, metadata


def process_file_list(
    files: List[Path],
    config: PreprocessConfig,
    desc: str = ""
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Process a list of audio files into spectrograms.

    Args:
        files: List of audio file paths
        config: Preprocessing configuration
        desc: Description for progress bar

    Returns:
        Tuple of:
            - Stacked spectrograms array of shape (N, n_mels, time_frames)
            - List of metadata dicts
    """
    all_specs = []
    all_meta = []

    for file_path in tqdm(files, desc=desc):
        specs, meta = process_file(file_path, config)
        all_specs.extend(specs)
        all_meta.extend(meta)

    if not all_specs:
        raise ValueError("No spectrogram windows were produced. Check input files and window configuration.")

    return np.stack(all_specs), all_meta


def save_processed_data(
    spectrograms: np.ndarray,
    metadata: List[Dict[str, Any]],
    output_dir: Path,
    name: str
) -> None:
    """
    Save processed spectrograms and metadata.

    Args:
        spectrograms: Array of spectrograms
        metadata: List of metadata dicts
        output_dir: Directory to save to
        name: Base name for the files
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save spectrograms as numpy file
    np.save(output_dir / f"{name}_spectrograms.npy", spectrograms)

    # Save metadata as JSON
    with open(output_dir / f"{name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved {name} data to {output_dir}")


def add_source_ids_to_metadata(
    metadata: List[Dict[str, Any]],
    files: List[Path]
) -> List[Dict[str, Any]]:
    """
    Add source_dir/source_id fields to cached metadata using the original file list.

    This lets old caches be migrated without recomputing spectrograms.
    """
    files = [Path(file_path) for file_path in files]
    files_by_name: Dict[str, List[Path]] = {}
    files_by_identity: Dict[Tuple[str, str], Path] = {}

    for file_path in files:
        files_by_name.setdefault(file_path.name, []).append(file_path)
        files_by_identity[(file_path.parent.name, file_path.name)] = file_path

    updated = []

    for item in metadata:
        if "source_id" in item and "source_dir" in item:
            updated.append(item)
            continue

        source_file = item.get("source_file")
        if source_file is None:
            raise KeyError("Cannot infer source_id for cached metadata entry without source_file")

        source_dir = item.get("source_dir")
        if source_dir is not None:
            file_path = files_by_identity.get((source_dir, source_file))
            if file_path is None:
                raise KeyError(f"Cannot infer source_id for cached metadata file: {source_dir}/{source_file}")
        else:
            matching_files = files_by_name.get(source_file, [])
            if not matching_files:
                raise KeyError(f"Cannot infer source_id for cached metadata file: {source_file}")
            if len(matching_files) > 1:
                choices = ", ".join(f"{path.parent.name}/{path.name}" for path in matching_files)
                raise ValueError(
                    f"Cannot infer source_id for basename-only metadata file '{source_file}' "
                    f"because multiple source files match: {choices}"
                )

            file_path = matching_files[0]

        source_dir = file_path.parent.name
        migrated = dict(item)
        migrated["source_dir"] = source_dir
        migrated["source_id"] = f"{source_dir}/{source_file}"
        updated.append(migrated)

    return updated


def load_processed_data(input_dir: Path, name: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load previously processed spectrograms and metadata.

    Args:
        input_dir: Directory containing saved data
        name: Base name of the files

    Returns:
        Tuple of spectrograms array and metadata list
    """

    input_dir = Path(input_dir)

    spectrograms = np.load(input_dir / f"{name}_spectrograms.npy")

    with open(input_dir / f"{name}_metadata.json", "r") as f:
        metadata = json.load(f)

    return spectrograms, metadata


def compute_db_percentiles(
    files: List[Path],
    config: PreprocessConfig,
    n_samples: int = 100,
    percentiles: List[float] = [1, 5, 95, 99],
    seed: Optional[int] = 42
) -> Dict[str, float]:
    """
    Compute dB value percentiles from a sample of audio files.

    Use this to determine appropriate db_min and db_max values for global normalization.
    Run this BEFORE processing the full dataset to set config.db_min/db_max.

    Args:
        files: List of audio file paths (will sample from these)
        config: Preprocessing configuration (uses use_fixed_ref setting)
        n_samples: Number of files to sample (default 100)
        percentiles: Which percentiles to compute
        seed: Seed for local file sampling. Set to None for non-deterministic sampling.

    Returns:
        Dict with percentile values and recommendations, e.g.:
        {'p1': -85.2, 'p5': -72.1, 'p95': -15.3, 'p99': -8.1,
         'min': -95.0, 'max': -5.0, 'recommended_db_min': -85.2, 'recommended_db_max': -8.1}
    """
    # Sample files if there are more than n_samples
    if len(files) > n_samples:
        rng = np.random.default_rng(seed)
        sample_files = rng.choice(files, size=n_samples, replace=False)
    else:
        sample_files = files

    all_db_values = []

    for file_path in tqdm(sample_files, desc="Computing dB percentiles"):
        # Load audio
        y, _ = librosa.load(file_path, sr=config.sr, mono=True)

        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=config.sr, n_fft=config.n_fft,
            hop_length=config.hop_length, n_mels=config.n_mels
        )

        # Convert to dB with same reference as will be used in processing
        if config.use_fixed_ref:
            S_db = librosa.power_to_db(S, ref=1.0)
        else:
            S_db = librosa.power_to_db(S, ref=np.max)

        all_db_values.extend(S_db.flatten())

    all_db_values = np.array(all_db_values)

    # Compute percentiles
    result = {
        'min': float(all_db_values.min()),
        'max': float(all_db_values.max()),
    }

    for p in percentiles:
        result[f'p{int(p)}'] = float(np.percentile(all_db_values, p))

    # Recommendations: use 1st percentile for db_min, 99th for db_max
    result['recommended_db_min'] = result.get('p1', result['min'])
    result['recommended_db_max'] = result.get('p99', result['max'])

    return result


def get_spectrogram_shape(config: PreprocessConfig) -> Tuple[int, int]:
    """
    Calculate the expected spectrogram shape for a given config.

    Useful for setting up model architectures.

    Args:
        config: Preprocessing configuration

    Returns:
        Tuple of (n_mels, time_frames) after power-of-2 padding
    """
    # librosa uses center=True by default, which pads the signal
    time_frames = 1 + config.window_samples // config.hop_length
    # Pad to next power of 2 (matching compute_mel_spectrogram)
    next_pow2 = 1 << (time_frames - 1).bit_length()
    return (config.n_mels, next_pow2)
