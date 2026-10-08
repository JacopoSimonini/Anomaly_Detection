import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class SplitIndices:
    """Container for dataset split indices."""
    ae_train: np.ndarray      # Indices for AE training
    ae_val: np.ndarray        # Indices for AE validation
    if_threshold: np.ndarray  # Indices for IF threshold tuning (normal + anomaly)
    final_test: np.ndarray    # Indices for final evaluation (normal + anomaly)


def split_file_list(
    files: List[Path],
    ratios: Tuple[float, ...],
    seed: int = 42
) -> List[List[Path]]:
    """
    Split a list of files into multiple subsets.

    Args:
        files: List of file paths
        ratios: Tuple of ratios (must sum to 1.0)
        seed: Random seed for reproducibility

    Returns:
        List of file lists, one per ratio
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {sum(ratios)}"

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(files))

    splits = []
    start = 0
    for i, ratio in enumerate(ratios):
        if i == len(ratios) - 1:
            # Last split gets remaining to avoid rounding issues
            end = len(files)
        else:
            end = start + int(len(files) * ratio)
        split_indices = indices[start:end]
        splits.append([files[idx] for idx in split_indices])
        start = end

    return splits


def create_data_splits(
    train_normal_dir: Path,
    test_normal_dir: Path,
    anomaly_dir: Path,
    ae_train_ratio: float = 0.8,
    if_threshold_normal_ratio: float = 0.5,
    if_threshold_anomaly_ratio: float = 0.3,
    seed: int = 42,
    file_extension: str = ".wav"
) -> Dict[str, List[Path]]:
    """
    Create all data splits at file level.

    Split strategy:
    - train-normal: 80% AE train, 20% AE val
    - test-normal: 50% IF threshold tuning, 50% final test
    - anomaly: 30% IF threshold tuning, 70% final test

    Args:
        train_normal_dir: Directory with training normal audio
        test_normal_dir: Directory with test normal audio
        anomaly_dir: Directory with anomaly audio
        ae_train_ratio: Fraction of train-normal for AE training
        if_threshold_normal_ratio: Fraction of test-normal for IF tuning
        if_threshold_anomaly_ratio: Fraction of anomaly for IF tuning
        seed: Random seed
        file_extension: Audio file extension

    Returns:
        Dictionary with keys:
            - 'ae_train': files for AE training
            - 'ae_val': files for AE validation
            - 'if_threshold_normal': normal files for IF tuning
            - 'if_threshold_anomaly': anomaly files for IF tuning
            - 'test_normal': normal files for final test
            - 'test_anomaly': anomaly files for final test
    """
    # Get file lists
    train_normal_files = sorted(Path(train_normal_dir).glob(f"*{file_extension}"))
    test_normal_files = sorted(Path(test_normal_dir).glob(f"*{file_extension}"))
    anomaly_files = sorted(Path(anomaly_dir).glob(f"*{file_extension}"))

    # Split train-normal for AE
    ae_train, ae_val = split_file_list(
        train_normal_files,
        (ae_train_ratio, 1 - ae_train_ratio),
        seed=seed
    )

    # Split test-normal for IF threshold vs final test
    if_normal, test_normal = split_file_list(
        test_normal_files,
        (if_threshold_normal_ratio, 1 - if_threshold_normal_ratio),
        seed=seed
    )

    # Split anomaly for IF threshold vs final test
    if_anomaly, test_anomaly = split_file_list(
        anomaly_files,
        (if_threshold_anomaly_ratio, 1 - if_threshold_anomaly_ratio),
        seed=seed
    )

    return {
        'ae_train': ae_train,
        'ae_val': ae_val,
        'if_threshold_normal': if_normal,
        'if_threshold_anomaly': if_anomaly,
        'test_normal': test_normal,
        'test_anomaly': test_anomaly
    }


def print_split_summary(splits: Dict[str, List[Path]]) -> None:
    """Print summary of data splits."""

    print("DATA SPLIT SUMMARY")

    print("\n[Autoencoder]")
    print(f"  Training:   {len(splits['ae_train']):>5} files")
    print(f"  Validation: {len(splits['ae_val']):>5} files")

    print("\n[Isolation Forest Training]")
    print(f"  Normal:     {(len(splits['ae_train']) + len(splits['ae_val'])):>5} files (latent representations)")
    print("\n[Isolation Forest Validation (and Tuning)]")
    print(f"  Normal:     {len(splits['if_threshold_normal']):>5} files")
    print(f"  Anomaly:    {len(splits['if_threshold_anomaly']):>5} files")

    print("\n[Final Test]")
    print(f"  Normal:     {len(splits['test_normal']):>5} files")
    print(f"  Anomaly:    {len(splits['test_anomaly']):>5} files")

class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for spectrograms.

    Wraps preprocessed spectrograms for use with DataLoader.
    """

    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: Optional[np.ndarray] = None,
        transform: Optional[callable] = None
    ):
        """
        Args:
            spectrograms: Array of shape (N, n_mels, time_frames)
            labels: Optional array of labels (0=normal, 1=anomaly)
            transform: Optional transform to apply to each spectrogram
        """
        self.spectrograms = spectrograms
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        spec = self.spectrograms[idx]

        # Add channel dimension: (n_mels, time) -> (1, n_mels, time)
        spec = spec[np.newaxis, :, :]

        if self.transform:
            spec = self.transform(spec)

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float()

        result = {'spectrogram': spec_tensor}

        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return result

