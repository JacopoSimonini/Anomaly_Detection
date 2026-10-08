import numpy as np
import pytest

from src.models.classical import (
    _make_group_cv,
    fit_svm_supervised,
    fit_xgboost,
    score_svm_supervised,
)


def _toy_grouped_data():
    X = np.array([
        [-2.0, -1.9], [-2.1, -2.0],
        [-1.0, -1.1], [-1.2, -1.0],
        [1.0, 1.1], [1.2, 1.0],
        [2.0, 2.1], [2.2, 2.0],
    ])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    groups = np.array([
        "normal_a", "normal_a",
        "normal_b", "normal_b",
        "anomaly_a", "anomaly_a",
        "anomaly_b", "anomaly_b",
    ])
    return X, y, groups


def test_group_cv_keeps_source_files_out_of_both_train_and_validation():
    _, y, groups = _toy_grouped_data()
    cv = _make_group_cv(y, groups, cv=2, random_state=42)

    for train_idx, val_idx in cv.split(np.zeros_like(y), y, groups):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups)


def test_fit_svm_supervised_requires_groups():
    X, y, _ = _toy_grouped_data()

    with pytest.raises(ValueError, match="groups is required"):
        fit_svm_supervised(X, y, groups=None)


def test_fit_svm_supervised_returns_pipeline_and_grouped_cv_info():
    X, y, groups = _toy_grouped_data()

    model, info = fit_svm_supervised(
        X,
        y,
        groups=groups,
        param_grid={"svc__C": [1.0], "svc__gamma": ["scale"]},
        cv=2,
        n_jobs=1,
    )
    scores = score_svm_supervised(model, X)

    assert info["grouped_cv"] is True
    assert info["cv_splits"] == 2
    assert info["n_groups"] == 4
    assert scores.shape == (len(X),)


def test_fit_xgboost_validates_groups_before_importing_optional_dependency():
    X, y, _ = _toy_grouped_data()

    with pytest.raises(ValueError, match="groups is required"):
        fit_xgboost(X, y, groups=None)
