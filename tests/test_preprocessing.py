from pathlib import Path

import numpy as np
import pytest

from src.config import PreprocessConfig
from src import preprocessing


def test_process_file_metadata_includes_stable_source_id(monkeypatch):
    config = PreprocessConfig(sr=4, window_sec=1.0, hop_sec=1.0)
    file_path = Path("data/raw/train-normal/example.wav")

    monkeypatch.setattr(
        preprocessing.librosa,
        "load",
        lambda path, sr, mono: (np.arange(8, dtype=np.float32), sr),
    )
    monkeypatch.setattr(
        preprocessing,
        "compute_mel_spectrogram",
        lambda segment, config: np.zeros((config.n_mels, 4), dtype=np.float32),
    )

    specs, metadata = preprocessing.process_file(file_path, config)

    assert len(specs) == 2
    assert metadata[0]["source_file"] == "example.wav"
    assert metadata[0]["source_dir"] == "train-normal"
    assert metadata[0]["source_id"] == "train-normal/example.wav"
    assert metadata[0]["window_idx"] == 0
    assert metadata[1]["start_sec"] == 1.0


def test_add_source_ids_to_metadata_migrates_old_cache_metadata():
    metadata = [{"source_file": "same.wav", "window_idx": 0}]
    files = [Path("data/raw/test-normal/same.wav")]

    migrated = preprocessing.add_source_ids_to_metadata(metadata, files)

    assert migrated[0]["source_file"] == "same.wav"
    assert migrated[0]["source_dir"] == "test-normal"
    assert migrated[0]["source_id"] == "test-normal/same.wav"
    assert migrated[0]["window_idx"] == 0


def test_add_source_ids_to_metadata_rejects_ambiguous_basenames():
    metadata = [{"source_file": "same.wav", "window_idx": 0}]
    files = [
        Path("data/raw/test-normal/same.wav"),
        Path("data/raw/anomaly/same.wav"),
    ]

    with pytest.raises(ValueError, match="multiple source files match"):
        preprocessing.add_source_ids_to_metadata(metadata, files)


def test_add_source_ids_to_metadata_uses_source_dir_to_disambiguate():
    metadata = [{"source_file": "same.wav", "source_dir": "anomaly", "window_idx": 0}]
    files = [
        Path("data/raw/test-normal/same.wav"),
        Path("data/raw/anomaly/same.wav"),
    ]

    migrated = preprocessing.add_source_ids_to_metadata(metadata, files)

    assert migrated[0]["source_dir"] == "anomaly"
    assert migrated[0]["source_id"] == "anomaly/same.wav"


def test_compute_db_percentiles_uses_local_rng(monkeypatch):
    files = [Path(f"file_{idx}.wav") for idx in range(5)]
    config = PreprocessConfig()

    def fake_load(path, sr, mono):
        value = int(Path(path).stem.split("_")[1]) + 1.0
        return np.array([value], dtype=np.float32), sr

    monkeypatch.setattr(preprocessing.librosa, "load", fake_load)
    monkeypatch.setattr(
        preprocessing.librosa.feature,
        "melspectrogram",
        lambda y, sr, n_fft, hop_length, n_mels: np.full((1, 1), y[0], dtype=np.float32),
    )
    monkeypatch.setattr(preprocessing.librosa, "power_to_db", lambda S, ref: S)

    np.random.seed(123)
    expected_next_global_random = np.random.random()
    np.random.seed(123)

    first = preprocessing.compute_db_percentiles(
        files,
        config,
        n_samples=2,
        percentiles=[50],
        seed=99,
    )
    second = preprocessing.compute_db_percentiles(
        files,
        config,
        n_samples=2,
        percentiles=[50],
        seed=99,
    )
    actual_next_global_random = np.random.random()

    assert first == second
    assert actual_next_global_random == expected_next_global_random
