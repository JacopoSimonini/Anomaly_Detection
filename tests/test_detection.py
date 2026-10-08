import numpy as np
import pytest

from src.detection import aggregate_scores_to_files, evaluate_detection, find_optimal_threshold


def test_aggregate_scores_uses_source_id_before_source_file():
    scores = np.array([0.1, 0.3, 0.8, 1.0])
    metadata = [
        {"source_file": "same.wav", "source_id": "train-normal/same.wav"},
        {"source_file": "same.wav", "source_id": "train-normal/same.wav"},
        {"source_file": "same.wav", "source_id": "test-normal/same.wav"},
        {"source_file": "same.wav", "source_id": "test-normal/same.wav"},
    ]

    file_scores, file_names = aggregate_scores_to_files(scores, metadata, strategy="mean")

    assert file_names == ["train-normal/same.wav", "test-normal/same.wav"]
    np.testing.assert_allclose(file_scores, [0.2, 0.9])


def test_aggregate_scores_falls_back_to_source_file_for_old_metadata():
    scores = np.array([0.2, 0.6])
    metadata = [{"source_file": "old.wav"}, {"source_file": "old.wav"}]

    file_scores, file_names = aggregate_scores_to_files(scores, metadata, strategy="max")

    assert file_names == ["old.wav"]
    np.testing.assert_allclose(file_scores, [0.6])


def test_aggregate_scores_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        aggregate_scores_to_files(
            np.array([0.1, 0.2]),
            [{"source_id": "train-normal/a.wav"}],
        )


def test_find_optimal_threshold_uses_score_boundaries_for_balanced_accuracy():
    normal = np.array([0.1, 0.2, 0.3])
    anomaly = np.array([0.8, 0.9, 1.0])

    threshold, metrics = find_optimal_threshold(normal, anomaly)

    assert 0.3 < threshold <= 0.8
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["tpr"] == 1.0
    assert metrics["tnr"] == 1.0


def test_find_optimal_threshold_rejects_empty_classes():
    with pytest.raises(ValueError, match="non-empty"):
        find_optimal_threshold(np.array([]), np.array([0.5]))


def test_evaluate_detection_reports_balanced_accuracy():
    results = evaluate_detection(
        scores_normal=np.array([0.1, 0.2, 0.9]),
        scores_anomaly=np.array([0.8, 1.0]),
        threshold=0.75,
    )

    assert "balanced_accuracy" in results
    assert results["balanced_accuracy"] == pytest.approx(5 / 6)
