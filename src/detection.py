"""
Anomaly detection utilities for latent space analysis.

This module provides functions for:
- Extracting latent representations from the ConvAE
- Aggregating window-level scores to file-level predictions
- Threshold optimization and evaluation
- Visualization utilities
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Literal, TYPE_CHECKING
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score
)

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader


# =============================================================================
# Latent Representation Extraction
# =============================================================================

def extract_latent_representations(
    model,
    dataloader: "DataLoader",
    device: "torch.device"
) -> np.ndarray:
    """
    Extract latent representations from the encoder for all samples.

    Note: @torch.no_grad() disables gradient computation (saves memory),
    while model.eval() changes BatchNorm/Dropout behavior. Both are needed.

    Args:
        model: Trained ConvAE model
        dataloader: DataLoader containing spectrograms
        device: Device to run inference on

    Returns:
        Array of shape (N, latent_dim) with latent vectors
    """
    import torch

    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrogram'].to(device)
            z = model.encode(x)
            latent_vectors.append(z.cpu().numpy())

    return np.concatenate(latent_vectors, axis=0)


def extract_reconstruction_errors(
    model,
    dataloader: "DataLoader",
    device: "torch.device"
) -> np.ndarray:
    """
    Compute reconstruction error (MSE) for all samples in a DataLoader.

    This wraps the model's reconstruction_error() method to process
    an entire DataLoader, avoiding repeated boilerplate in notebooks.

    Args:
        model: Trained ConvAE model
        dataloader: DataLoader containing spectrograms
        device: Device to run inference on

    Returns:
        Array of shape (N,) with reconstruction errors per sample
    """
    import torch

    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch['spectrogram'].to(device)
            error = model.reconstruction_error(x, reduction='mean')
            errors.append(error.cpu().numpy())

    return np.concatenate(errors, axis=0)


# =============================================================================
# Score Aggregation (Window -> File level)
# =============================================================================

def aggregate_scores_to_files(
    window_scores: np.ndarray,
    metadata: List[Dict],
    strategy: Literal['max', 'mean', 'median', 'percentile_90'] = 'max',
    group_key: str = "source_id"
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate window-level anomaly scores to file-level scores.

    Each audio file produces multiple spectrogram windows (~9 per 10s file with the default setup).
    Since ground truth labels are at the file level, window scores must be
    aggregated into a single score per file.

    Strategies:
        - 'max': File score = max window score. Conservative approach that
          flags a file if ANY window is anomalous. Good for detecting
          localized/transient anomalies.
        - 'mean': Average score across windows. Smooths out noise but may
          miss brief anomalies.
        - 'median': Robust to outlier windows.
        - 'percentile_90': Balance between max and mean.

    Args:
        window_scores: Array of shape (N_windows,) with per-window scores
        metadata: List of dicts with source identity for each window
        strategy: Aggregation method
        group_key: Metadata key used to group windows. Defaults to 'source_id',
            falling back to 'source_file' for backward compatibility with old caches.

    Returns:
        Tuple of:
            - file_scores: Array of shape (N_files,) with aggregated scores
            - file_names: List of file names in same order
    """
    if len(window_scores) != len(metadata):
        raise ValueError(f"window_scores and metadata must have the same length, got {len(window_scores)} and {len(metadata)}")

    if len(window_scores) == 0:
        raise ValueError("Cannot aggregate an empty score array")

    # Group scores by source file
    file_to_scores = {}
    for score, meta in zip(window_scores, metadata):
        source = meta.get(group_key) or meta.get('source_file')
        if source is None:
            raise KeyError(f"Metadata entry must contain '{group_key}' or 'source_file'")
        if source not in file_to_scores:
            file_to_scores[source] = []
        file_to_scores[source].append(score)

    # Aggregate scores per file
    file_names = list(file_to_scores.keys())
    file_scores = []

    for fname in file_names:
        scores = np.array(file_to_scores[fname])

        if strategy == 'max':
            agg_score = np.max(scores)
        elif strategy == 'mean':
            agg_score = np.mean(scores)
        elif strategy == 'median':
            agg_score = np.median(scores)
        elif strategy == 'percentile_90':
            agg_score = np.percentile(scores, 90)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

        file_scores.append(agg_score)

    return np.array(file_scores), file_names


# =============================================================================
# Threshold Optimization
# =============================================================================

def find_optimal_threshold(
    scores_normal: np.ndarray,
    scores_anomaly: np.ndarray,
    metric: Literal['f1', 'accuracy', 'balanced_accuracy'] = 'balanced_accuracy',
    n_thresholds: Optional[int] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal threshold for anomaly detection via grid search.

    Args:
        scores_normal: Anomaly scores for normal samples (should be lower)
        scores_anomaly: Anomaly scores for anomaly samples (should be higher)
        metric: Optimization metric:
            - 'f1': F1 score (harmonic mean of precision and recall)
            - 'accuracy': Overall accuracy
            - 'balanced_accuracy': Average of TPR and TNR (handles class imbalance)
        n_thresholds: Optional cap on threshold candidates. By default every
            score-derived decision boundary is evaluated.

    Returns:
        Tuple of:
            - optimal_threshold: Best threshold value
            - metrics_at_threshold: Dict with metrics at optimal threshold
    """
    # Create labels: 0 = normal, 1 = anomaly
    y_true = np.concatenate([
        np.zeros(len(scores_normal)),
        np.ones(len(scores_anomaly))
    ])
    scores = np.concatenate([scores_normal, scores_anomaly])

    if len(scores_normal) == 0 or len(scores_anomaly) == 0:
        raise ValueError("Both normal and anomaly score arrays must be non-empty")

    # Evaluate score-derived decision boundaries instead of a coarse linspace.
    # With rule score >= threshold, midpoints cover every possible ranking split.
    unique_scores = np.unique(scores)
    if len(unique_scores) == 1:
        eps = max(abs(float(unique_scores[0])) * 1e-9, 1e-12)
        thresholds = np.array([unique_scores[0] - eps, unique_scores[0], unique_scores[0] + eps])
    else:
        midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2
        eps = max(float(np.ptp(unique_scores)) * 1e-9, 1e-12)
        thresholds = np.concatenate([
            [unique_scores[0] - eps],
            unique_scores,
            midpoints,
            [unique_scores[-1] + eps]
        ])

    if n_thresholds is not None and n_thresholds > 0 and len(thresholds) > n_thresholds:
        idx = np.linspace(0, len(thresholds) - 1, n_thresholds, dtype=int)
        thresholds = thresholds[np.unique(idx)]

    best_threshold = None
    best_metric_value = -np.inf
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)

        # Compute metrics
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tpr = recall  # True positive rate (sensitivity)
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate (specificity)
        balanced_acc = (tpr + tnr) / 2

        # Select metric for optimization
        if metric == 'f1':
            metric_value = f1
        elif metric == 'accuracy':
            metric_value = accuracy
        elif metric == 'balanced_accuracy':
            metric_value = balanced_acc
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = thresh
            best_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tpr': tpr,
                'tnr': tnr,
                'balanced_accuracy': balanced_acc
            }

    return best_threshold, best_metrics


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_detection(
    scores_normal: np.ndarray,
    scores_anomaly: np.ndarray,
    threshold: float
) -> Dict:
    """
    Evaluate anomaly detection performance.

    Args:
        scores_normal: Anomaly scores for normal test samples
        scores_anomaly: Anomaly scores for anomaly test samples
        threshold: Decision threshold (score >= threshold -> anomaly)

    Returns:
        Dict with all metrics and data for plotting:
            - accuracy, precision, recall, f1: Classification metrics
            - auc_roc, auc_pr: Area under ROC and PR curves
            - confusion_matrix: 2x2 confusion matrix
            - threshold: The threshold used
            - fpr, tpr, roc_thresholds: Data for ROC curve
            - precision_curve, recall_curve: Data for PR curve
    """
    # Create labels and predictions
    y_true = np.concatenate([
        np.zeros(len(scores_normal)),
        np.ones(len(scores_anomaly))
    ])
    scores = np.concatenate([scores_normal, scores_anomaly])

    if len(scores_normal) == 0 or len(scores_anomaly) == 0:
        raise ValueError("Both normal and anomaly score arrays must be non-empty")
    y_pred = (scores >= threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC (threshold-independent)
    auc_roc = roc_auc_score(y_true, scores)
    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)

    # Precision-Recall curve and AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'confusion_matrix': cm,
        'threshold': threshold,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': roc_thresholds,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_score_distributions(
    scores_normal: np.ndarray,
    scores_anomaly: np.ndarray,
    threshold: Optional[float] = None,
    title: str = "Anomaly Score Distribution",
    ax=None
):
    """
    Plot histograms of anomaly scores for normal and anomaly samples.

    Args:
        scores_normal: Scores for normal samples
        scores_anomaly: Scores for anomaly samples
        threshold: Optional threshold line to display
        title: Plot title
        ax: Optional matplotlib axes

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Plot histograms
    bins = 50
    ax.hist(scores_normal, bins=bins, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(scores_anomaly, bins=bins, alpha=0.6, label='Anomaly', color='red', density=True)

    if threshold is not None:
        ax.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.4f})')

    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_roc_curve(results: Dict, method_name: str = "", ax=None):
    """
    Plot ROC curve with AUC.

    Args:
        results: Dict from evaluate_detection
        method_name: Name for the legend
        ax: Optional matplotlib axes

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    label = f'{method_name} (AUC = {results["auc_roc"]:.4f})' if method_name else f'AUC = {results["auc_roc"]:.4f}'
    ax.plot(results['fpr'], results['tpr'], linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    return ax


def plot_precision_recall_curve(results: Dict, method_name: str = "", ax=None):
    """
    Plot Precision-Recall curve with AUC.

    Args:
        results: Dict from evaluate_detection
        method_name: Name for the legend
        ax: Optional matplotlib axes

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    label = f'{method_name} (AUC = {results["auc_pr"]:.4f})' if method_name else f'AUC = {results["auc_pr"]:.4f}'
    ax.plot(results['recall_curve'], results['precision_curve'], linewidth=2, label=label)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    return ax


def plot_confusion_matrix(results: Dict, method_name: str = "", ax=None):
    """
    Plot confusion matrix as heatmap.

    Args:
        results: Dict from evaluate_detection
        method_name: Title prefix
        ax: Optional matplotlib axes

    Returns:
        matplotlib axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    cm = results['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=['Normal', 'Anomaly'],
        yticklabels=['Normal', 'Anomaly'],
        ylabel='True Label',
        xlabel='Predicted Label'
    )

    title = f'{method_name} Confusion Matrix' if method_name else 'Confusion Matrix'
    ax.set_title(title)

    # Add text annotations
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=14)

    return ax


def plot_comparison_roc(
    results_dict: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for multiple methods on the same axes.

    Args:
        results_dict: Dict mapping method names to result dicts
        save_path: Optional path to save the figure

    Returns:
        Tuple of (figure, axes)
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(results['fpr'], results['tpr'], linewidth=2, color=color,
                label=f'{name} (AUC = {results["auc_roc"]:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def create_comparison_table(results_dict: Dict[str, Dict]) -> str:
    """
    Create a formatted comparison table of detection results.

    Args:
        results_dict: Dict mapping method names to result dicts

    Returns:
        Formatted string table
    """
    header = f"{'Method':<25} {'Accuracy':>10} {'Bal Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10} {'AUC-PR':>10}"
    separator = "-" * len(header)

    rows = [header, separator]

    for name, results in results_dict.items():
        row = f"{name:<25} {results['accuracy']:>10.4f} {results['balanced_accuracy']:>10.4f} {results['precision']:>10.4f} {results['recall']:>10.4f} {results['f1']:>10.4f} {results['auc_roc']:>10.4f} {results['auc_pr']:>10.4f}"
        rows.append(row)

    return "\n".join(rows)


def print_results(results: Dict, method_name: str = ""):
    """Print formatted detection results."""
    header = f" {method_name} Results " if method_name else " Detection Results "
    print(f"\n{'='*50}")
    print(f"{header:=^50}")
    print(f"{'='*50}")
    print(f"  Threshold:    {results['threshold']:.6f}")
    print(f"  Accuracy:     {results['accuracy']:.4f}")
    print(f"  Bal Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"  Precision:    {results['precision']:.4f}")
    print(f"  Recall:       {results['recall']:.4f}")
    print(f"  F1 Score:     {results['f1']:.4f}")
    print(f"  AUC-ROC:      {results['auc_roc']:.4f}")
    print(f"  AUC-PR:       {results['auc_pr']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                 Pred Normal  Pred Anomaly")
    cm = results['confusion_matrix']
    print(f"    True Normal      {cm[0,0]:>5}         {cm[0,1]:>5}")
    print(f"    True Anomaly     {cm[1,0]:>5}         {cm[1,1]:>5}")
    print(f"{'='*50}\n")
