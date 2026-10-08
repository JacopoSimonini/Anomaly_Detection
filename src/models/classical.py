"""
Classical anomaly classifiers built on top of (or alongside) the frozen ConvAE.

Provides four downstream methods that complement the existing IF / LOF detectors:
    1. fit_oc_svm           - One-Class SVM, unsupervised on AE latents of normal data
    2. fit_svm_supervised   - Supervised SVM-RBF on labelled if_threshold data
    3. fit_xgboost          - Supervised XGBoost classifier
    4. compute_mfcc_features - MFCC-stat feature extractor for an AE-bypassing baseline

All scoring functions return arrays where HIGHER score = MORE anomalous, so they plug
directly into find_optimal_threshold / evaluate_detection from src.detection.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, OneClassSVM

from ..config import PreprocessConfig


# =============================================================================
# 1. One-Class SVM (unsupervised, normal-only)
# =============================================================================

def fit_oc_svm(
    X_train_normal: np.ndarray,
    nu: float = 0.1,
    gamma: str = "scale",
    kernel: str = "rbf",
) -> Tuple[OneClassSVM, StandardScaler]:
    """
    Fit a One-Class SVM on AE latent vectors of normal data.

    Args:
        X_train_normal: (N, D) latent vectors of normal training samples
        nu: upper bound on fraction of training errors (controls boundary tightness;
            NOT a contamination estimate since training data is pure normal)
        gamma: RBF kernel coefficient
        kernel: SVM kernel

    Returns:
        (fitted OneClassSVM, fitted StandardScaler) - scaler must be re-applied at scoring time
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_normal)

    model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    model.fit(X_scaled)
    return model, scaler


def score_oc_svm(model: OneClassSVM, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """
    Score samples with a fitted One-Class SVM. Higher = more anomalous.

    OneClassSVM.decision_function returns positive for inliers and negative for outliers,
    so the sign is flipped to follow the project convention (higher = more anomalous).
    """
    X_scaled = scaler.transform(X)
    return -model.decision_function(X_scaled)


# =============================================================================
# 2. Supervised SVM (RBF) with k-fold CV hyperparameter selection
# =============================================================================

DEFAULT_SVM_GRID: Dict[str, List] = {
    "svc__C": [0.1, 1.0, 10.0],
    "svc__gamma": ["scale", "auto"],
}


def _validate_groups(groups: np.ndarray, n_samples: int) -> np.ndarray:
    """Validate grouped CV labels for window-level supervised training."""
    if groups is None:
        raise ValueError("groups is required for supervised window-level CV to avoid file leakage")

    groups = np.asarray(groups)
    if groups.shape[0] != n_samples:
        raise ValueError(f"groups must have length {n_samples}, got {groups.shape[0]}")

    if len(np.unique(groups)) < 2:
        raise ValueError("groups must contain at least two distinct source files")

    return groups


def _make_group_cv(y: np.ndarray, groups: np.ndarray, cv: int, random_state: int) -> StratifiedGroupKFold:
    """Create a StratifiedGroupKFold with a feasible number of splits."""
    groups = _validate_groups(groups, len(y))
    y = np.asarray(y, dtype=int)
    groups_per_class = [len(np.unique(groups[y == cls])) for cls in np.unique(y)]
    max_class_splits = min(groups_per_class) if groups_per_class else 0
    max_group_splits = len(np.unique(groups))
    n_splits = min(cv, max_class_splits, max_group_splits)

    if n_splits < 2:
        raise ValueError("Need at least two source-file groups per class for grouped CV")

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def fit_svm_supervised(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[Pipeline, Dict]:
    """
    Fit a supervised SVM-RBF with stratified grouped k-fold grid search.

    Class imbalance handled with class_weight='balanced'. probability=True so the model can
    return predict_proba scores. Scaling is inside the Pipeline so it is fit only on
    each training fold during CV.

    Args:
        X: (N, D) features
        y: (N,) binary labels (0 = normal, 1 = anomaly)
        groups: (N,) source-file identifiers. Required to keep windows from the same
            file in the same CV fold.
        param_grid: dict of hyperparameter lists; defaults to DEFAULT_SVM_GRID
        cv: number of stratified folds
        scoring: sklearn scoring metric for grid search
        random_state: for the inner KFold splitter
        n_jobs: parallel jobs in grid search

    Returns:
        (fitted Pipeline, best_params/info)
    """
    if param_grid is None:
        param_grid = DEFAULT_SVM_GRID

    groups = _validate_groups(groups, len(y))

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)),
    ])
    sgkf = _make_group_cv(y, groups, cv=cv, random_state=random_state)
    grid = GridSearchCV(base, param_grid, cv=sgkf, scoring=scoring, n_jobs=n_jobs, refit=True)
    grid.fit(X, y, groups=groups)

    return grid.best_estimator_, {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "cv_splits": grid.n_splits_,
        "n_groups": int(len(np.unique(groups))),
        "grouped_cv": True,
        "scoring": scoring,
    }


def score_svm_supervised(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return P(anomaly) so higher = more anomalous."""
    return model.predict_proba(X)[:, 1]


# =============================================================================
# 3. XGBoost (supervised)
# =============================================================================

DEFAULT_XGB_GRID: Dict[str, List] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1],
}


def fit_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    n_estimators: int = 200,
    subsample: float = 0.8,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = 42,
    n_jobs: int = 1,
) -> Tuple["XGBClassifier", Dict]:  # noqa: F821 - quoted for optional dep
    """
    Fit an XGBoost classifier with stratified grouped k-fold grid search.

    Class imbalance handled via scale_pos_weight = n_neg / n_pos.

    Returns:
        (fitted XGBClassifier, info dict)
    """
    groups = _validate_groups(groups, len(y))

    from xgboost import XGBClassifier

    if param_grid is None:
        param_grid = DEFAULT_XGB_GRID

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    spw = n_neg / max(n_pos, 1)

    base = XGBClassifier(
        n_estimators=n_estimators,
        subsample=subsample,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        random_state=random_state,
        n_jobs=-1,  # XGBoost handles its own thread parallelism
        tree_method="hist",
        verbosity=0,
    )

    sgkf = _make_group_cv(y, groups, cv=cv, random_state=random_state)
    grid = GridSearchCV(base, param_grid, cv=sgkf, scoring=scoring, n_jobs=n_jobs, refit=True)
    grid.fit(X, y, groups=groups)

    return grid.best_estimator_, {
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "cv_splits": grid.n_splits_,
        "n_groups": int(len(np.unique(groups))),
        "grouped_cv": True,
        "scale_pos_weight": spw,
        "scoring": scoring,
    }


def score_xgboost(model, X: np.ndarray) -> np.ndarray:
    """Return P(anomaly) so higher = more anomalous."""
    return model.predict_proba(X)[:, 1]


# =============================================================================
# 4. MFCC-stat features (AE-bypassing baseline)
# =============================================================================

def compute_mfcc_features(
    mel_norm: np.ndarray,
    n_mfcc: int = 13,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """
    Compute MFCC-statistic features from a single mel spectrogram window.

    Operates on the cached, normalized log-mel spectrograms produced by
    src.preprocessing.compute_mel_spectrogram. Those spectrograms are in [0, 1] under
    the project's global-norm dB scheme; that mapping is inverted using the provided
    PreprocessConfig (db_min / db_max) so the input to librosa.feature.mfcc is on a
    proper dB scale.

    Note: MFCC is a DCT applied to the log-mel matrix, so any affine rescaling only
    affects the absolute scale of the coefficients (downstream StandardScaler in
    fit_svm_supervised normalizes anyway). Using the real bounds is a correctness nicety,
    not a performance lever.

    Returns a 4 * n_mfcc vector: [mean(MFCC), std(MFCC), mean(deltaMFCC), std(deltaMFCC)].

    Args:
        mel_norm: (n_mels, n_frames) normalized log-mel spectrogram (output of compute_mel_spectrogram)
        n_mfcc: number of MFCC coefficients to compute
        config: PreprocessConfig used to produce mel_norm (defaults to PreprocessConfig())

    Returns:
        (4 * n_mfcc,) feature vector
    """
    import librosa

    if config is None:
        config = PreprocessConfig()

    S_db = mel_norm.astype(np.float32) * (config.db_max - config.db_min) + config.db_min

    mfcc = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)

    feats = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        delta.mean(axis=1),
        delta.std(axis=1),
    ])
    return feats.astype(np.float32)


def compute_mfcc_features_batch(
    mel_specs: np.ndarray,
    n_mfcc: int = 13,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """
    Apply compute_mfcc_features to a batch of windows.

    Args:
        mel_specs: (N, n_mels, n_frames) array of normalized log-mel spectrograms
        n_mfcc: number of MFCC coefficients
        config: PreprocessConfig used to produce mel_specs

    Returns:
        (N, 4 * n_mfcc) feature matrix
    """
    return np.stack([compute_mfcc_features(m, n_mfcc=n_mfcc, config=config) for m in mel_specs])
