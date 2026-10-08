# Anomaly Detection for Water Pump Audio

from .detection import (
    # Latent extraction
    extract_latent_representations,
    extract_reconstruction_errors,
    # Aggregation
    aggregate_scores_to_files,
    # Threshold optimization
    find_optimal_threshold,
    # Evaluation
    evaluate_detection,
    print_results,
    # Visualization
    plot_score_distributions,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_comparison_roc,
    create_comparison_table
)
