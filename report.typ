#set document(
  title: "Audio Anomaly Detection on Water Pumps Using ConvAE Latent Representations",
)
#set page(
  paper: "a4",
  margin: (x: 1.65cm, y: 1.75cm),
  numbering: "1",
)
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#set figure(numbering: "1.")

#let run-id = "20260620_212421"

#align(center)[
  #text(size: 18pt, weight: "bold")[
    Audio Anomaly Detection on Water Pumps
  ]

  #text(size: 18pt, weight: "bold")[
    Using ConvAE Latent Representations
  ]

  #v(0.4em)
  #text(size: 11pt)[Digital Forensics and Biometrics - University project]

  #v(0.2em)
  #text(size: 9pt)[Canonical notebook run: #run-id]
]

#v(0.8em)

= Abstract

This work addresses anomaly detection in water-pump audio under the assignment requirement of first learning a latent representation and then detecting non-nominal samples in that feature space. The pipeline converts audio into fixed log-mel spectrogram windows, trains a convolutional autoencoder (ConvAE) on normal audio only, and evaluates several anomaly detectors on the resulting 64-dimensional latent representation. Among the assignment-aligned methods, ConvAE plus Local Outlier Factor (LOF) obtains the best result, with balanced accuracy 0.733 and AUC-ROC 0.788 on the held-out file-level test set. Supervised baselines perform better, especially an MFCC-statistics SVM with balanced accuracy 0.854, but these models use anomaly labels during training and are therefore interpreted as diagnostic baselines rather than as the main anomaly-detection result.

= Dataset and Experimental Protocol

The raw dataset is organized into three folders:

```text
data/raw/
  train-normal/
  test-normal/
  anomaly/
```

All splits are deterministic and file-level. This is necessary because each 10-second audio file is later expanded into overlapping windows, while the ground-truth label remains attached to the whole source file. Keeping file identity explicit avoids treating multiple windows from the same recording as independent files during aggregation and validation.

#figure(
  table(
    columns: (2.4fr, 1.2fr, 1.2fr),
    align: (left, right, right),
    table.header([Split], [Normal files], [Anomaly files]),
    [ConvAE train], [1792], [-],
    [ConvAE validation], [449], [-],
    [Threshold tuning], [200], [136],
    [Final test], [200], [320],
  ),
  caption: [File-level split used by the final notebook run.]
)

The ConvAE is trained only on normal audio from `train-normal`. The threshold-tuning split is used to choose decision thresholds, and the final test split is reserved for final metrics. One important dataset characteristic is that the ConvAE training and validation normal sets contain machine IDs `00`, `02`, and `04`, while the final normal set also contains ID `06`. Therefore, the unsupervised methods must handle both anomalies and unseen normal machine variation. This point is relevant for the false-positive behavior discussed later.

The split design separates three roles. The `train-normal` folder defines nominal behavior for representation learning. The threshold split calibrates the decision threshold once anomaly labels are available. The final test split estimates generalization after the threshold and hyperparameters have already been fixed. In this protocol, final test labels do not influence the reported threshold or model selection.

#figure(
  table(
    columns: (1.8fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, right, right, right, right),
    table.header([Split], [ID 00], [ID 02], [ID 04], [ID 06]),
    [AE train], [733], [719], [340], [0],
    [AE validation], [173], [186], [90], [0],
    [Threshold normal], [45], [51], [51], [53],
    [Threshold anomaly], [35], [38], [30], [33],
    [Final normal], [55], [49], [49], [47],
    [Final anomaly], [108], [73], [70], [69],
  ),
  caption: [Machine-ID composition by split. ID 06 is absent from ConvAE training but appears in threshold and final data.]
)

= Preprocessing

Every audio file is converted into a uniform time-frequency representation before entering the model. The preprocessing is the same for train, threshold-tuning, and final-test splits.

Each file is loaded as mono audio at 16 kHz. It is split into 2-second windows with a 1-second hop, so each 10-second recording produces 9 windows. For every window, the pipeline computes a mel spectrogram with `n_fft = 1024`, STFT `hop_length = 512`, `n_mels = 128`, and frequency range from 0 Hz to the 8 kHz Nyquist limit.

The power mel spectrogram is converted to decibels with `ref = 1.0`. This fixed reference is intentional: using each spectrogram's maximum as the reference would erase absolute intensity differences across files. The dB values are clipped to the 1st and 99th training percentiles, `-46.13 dB` and `-9.99 dB`, then linearly mapped to `[0, 1]`. Finally, the time axis is padded to 64 frames so the convolutional encoder and decoder have clean dimensions.

#figure(
  image("report_assets/db_distribution_fixed_reference.png", width: 100%),
  caption: [Training dB distribution under fixed-reference conversion. The percentile bounds define the global normalization range used for every split.]
)

The processed metadata stores `source_file`, `source_dir`, `source_id`, `window_idx`, `start_sec`, and `end_sec`. The stable `source_id` has the form `raw-subfolder/filename`, so duplicate basenames across folders cannot collide during file-level aggregation. The cached spectrograms were also checked against fresh preprocessing with the current configuration; representative files from every split matched exactly, so the final results are not caused by stale caches.

This preprocessing choice keeps the representation local rather than describing a 10-second clip as a single long image. The windowed view increases the number of training samples and reduces the chance that short faults are averaged away before feature extraction. At evaluation time, the windows are recombined at file level because the dataset labels are file-level.

= ConvAE Latent Representation

The representation model is a convolutional autoencoder trained on normal windows only. Its input is a normalized spectrogram of shape `1 x 128 x 64`. The encoder uses four convolutional stages with channels `32 -> 64 -> 128 -> 256`. The first encoder layer uses stride `(1, 2)` rather than `(2, 2)` so that frequency detail is preserved in the earliest stage. The flattened bottleneck is projected to a 64-dimensional latent vector. The decoder mirrors the encoder with transposed convolutions and ends with a sigmoid because the target spectrograms are normalized to `[0, 1]`.

The model is trained with mean squared reconstruction error, Adam with learning rate `1e-3`, weight decay `1e-5`, validation-based early stopping, and a reduce-on-plateau learning-rate scheduler. The final checkpoint trained for 52 epochs, with best validation MSE `0.007303` at epoch 50.

#figure(
  image("report_assets/convae_training_history.png", width: 82%),
  caption: [ConvAE training history. Train and validation losses converge smoothly, so the weak anomaly separation is not explained by an obvious training failure.]
)

After training, the ConvAE is frozen. Downstream methods either use the reconstruction error directly or consume the 64-dimensional latent vector. This separation matches the assignment structure: the ConvAE supplies the latent feature space, and different anomaly detectors are compared on top of the same representation.

The compression is substantial: each input window has `128 * 64 = 8192` spectrogram values, while the latent vector has 64 values. The bottleneck therefore forces the model to retain a compact summary of normal spectrogram structure. This is the representation used by Isolation Forest, LOF, OC-SVM, SVM, and XGBoost.

= Anomaly Detection Methods

The methods are divided by what information they are allowed to use.

#strong[Assignment-aligned unsupervised methods.] Reconstruction error uses the per-window MSE between an input spectrogram and its reconstruction. Isolation Forest is trained on the ConvAE latents of the normal train and validation windows; its score is sign-flipped so higher means more anomalous. LOF is also trained on normal ConvAE latents, with `novelty=True`, and detects samples in locally low-density regions. One-Class SVM is included as an additional one-class boundary method on standardized ConvAE latents.

#strong[Supervised diagnostic baselines.] SVM-RBF and XGBoost are trained on ConvAE latents from the threshold split, where anomaly labels are available. A second SVM-RBF baseline uses 52 MFCC-statistics features instead of the ConvAE latent space: mean and standard deviation of 13 MFCC coefficients and their first-order deltas. These models are not directly comparable to the unsupervised methods because they consume anomaly labels during training. They are included to test whether discriminative information is present in the audio independently of the one-class setting.

#strong[PANNs baseline.] PANNs was considered as an optional external feature-extractor baseline. It is intentionally not part of the final canonical result table because the latest notebook run has `RUN_PANNS_BASELINE = False`. The dependency and checkpoint requirements are heavier, and the assignment's central object is the ConvAE latent representation. If a future PANNs run is produced, it can be added as an appendix-level comparison rather than as a core method.

#figure(
  text(size: 8pt)[
  #table(
    columns: (1.7fr, 1.4fr, 1.45fr, 3.0fr),
    align: (left, left, left, left),
    inset: 4pt,
    table.header([Method group], [Feature input], [Uses anomaly labels?], [Role in the report]),
    [Reconstruction error], [ConvAE output], [No], [Direct AE anomaly score],
    [IF / LOF / OC-SVM], [ConvAE latent], [No], [Assignment-aligned anomaly detection on learned feature space],
    [SVM / XGBoost], [ConvAE latent], [Yes], [Diagnostic baseline for discriminative information in the latent space],
    [MFCC SVM], [MFCC statistics], [Yes], [Hand-crafted supervised baseline independent from ConvAE],
    [PANNs], [External embedding], [No for detector], [Optional external-feature baseline],
  )
  ],
  caption: [Method groups and fairness boundaries. The report's main claim should use the unsupervised ConvAE-based rows.]
)

= Thresholding and Evaluation

Every detector first produces one anomaly score per 2-second window. Since labels are defined at file level, scores are aggregated by `source_id`; the final protocol uses mean aggregation. Thresholds are selected only on the threshold-tuning split, optimizing balanced accuracy. The final test set remains untouched until the final evaluation.

Balanced accuracy is the default threshold-selection metric because the final test set is imbalanced: 200 normal files and 320 anomaly files. It averages specificity and recall, preventing plain accuracy from hiding a model that performs poorly on one class. AUC-ROC and AUC-PR are also reported because they are threshold-independent. In contrast, confusion matrices, precision, recall, F1, accuracy, and balanced accuracy depend on the chosen threshold.

For a chosen threshold, scores greater than or equal to the threshold are predicted as anomaly. Threshold candidates are derived from the actual score values rather than from a coarse fixed grid. This matters for small file-level validation sets because the best operating point can lie between two observed scores. The optimization target is:

#align(center)[$
  "balanced accuracy" = 1 / 2 ("TP" / ("TP" + "FN") + "TN" / ("TN" + "FP")).
$]

The first term is anomaly recall; the second term is normal recall. This makes the selected threshold a compromise between missed anomalies and false alarms.

= Implementation and Reproducibility

The canonical implementation is the notebook `anomaly_detection.ipynb`, supported by reusable modules under `src/`. The notebook is structured around one run protocol: load or preprocess spectrogram caches, load or train the ConvAE checkpoint, extract latents, train downstream detectors, tune thresholds, evaluate on final files, and save figures/tables/metrics.

The run used in this report is `outputs/notebook_runs/20260620_212421/`. It saved the plots included here, plus JSON and CSV metric snapshots. Those artifacts are the source of truth for every numeric value in the report. PANNs was disabled in this canonical run, so no PANNs result is included in the final comparison table.

Several implementation details support the validity of the reported comparison:

- metadata migration adds `source_id` to old caches without recomputing spectrograms;
- file-level aggregation groups by `source_id`, falling back to `source_file` only for older cache compatibility;
- supervised window-level cross-validation uses grouped folds so windows from the same file do not split across CV train and validation folds;
- SVM scaling is inside the scikit-learn pipeline, so each fold fits the scaler only on its training portion;
- synthetic tests cover metadata identity, aggregation, threshold search, and grouped supervised helpers.

These checks do not remove all statistical caveats. In particular, supervised baselines are trained and thresholded using the threshold split, so their tuning metrics are optimistic. The final test remains held out, but the supervised numbers are still best read as diagnostic comparisons rather than as the main anomaly-detection claim.

= Results

All results below are file-level metrics on the final test split, using mean aggregation and balanced-accuracy threshold selection.

#figure(
  text(size: 7.6pt)[
    #table(
      columns: (2.35fr, 0.62fr, 0.72fr, 0.66fr, 0.62fr, 0.62fr, 0.78fr, 0.72fr),
      align: (left, right, right, right, right, right, right, right),
      inset: 4pt,
      table.header([Method], [Acc], [Bal Acc], [Prec], [Rec], [F1], [AUC-ROC], [AUC-PR]),
      [ConvAE Reconstruction Error], [0.575], [0.595], [0.718], [0.509], [0.596], [0.646], [0.722],
      [ConvAE + IF], [0.563], [0.560], [0.669], [0.575], [0.618], [0.601], [0.701],
      [ConvAE + LOF], [0.723], [0.733], [0.831], [0.691], [0.754], [0.788], [0.852],
      [OC-SVM (AE latent)], [0.577], [0.545], [0.648], [0.684], [0.666], [0.615], [0.715],
      [SVM-sup (AE latent)], [0.788], [0.827], [0.995], [0.659], [0.793], [0.953], [0.971],
      [XGBoost (AE latent)], [0.792], [0.827], [0.977], [0.678], [0.801], [0.941], [0.965],
      [SVM-sup (MFCC stats)], [0.842], [0.854], [0.931], [0.803], [0.862], [0.934], [0.959],
    )
  ],
  caption: [Final file-level test metrics. The best assignment-aligned method is ConvAE + LOF; the best supervised diagnostic baseline is SVM on MFCC statistics.]
)

Among the unsupervised ConvAE-based methods, LOF is the clear winner. Its confusion matrix is `TN = 155`, `FP = 45`, `FN = 99`, `TP = 221`, corresponding to balanced accuracy 0.733. Reconstruction error alone is much weaker, and Isolation Forest is close to chance-level balanced accuracy. This indicates that simple reconstruction MSE is not sufficiently discriminative, while local density in the latent space captures more useful structure.

The supervised models perform better. SVM on MFCC statistics reaches balanced accuracy 0.854, with `TN = 181`, `FP = 19`, `FN = 63`, `TP = 257`. This indicates that the dataset contains detectable discriminative information, but the improvement comes from using anomaly labels and should not be interpreted as an unsupervised anomaly-detection result.

#figure(
  image("report_assets/convae_final_score_distributions.png", width: 100%),
  caption: [Final-test score distributions for the three main ConvAE methods. Reconstruction error and Isolation Forest show strong overlap; LOF separates normal and anomaly files more clearly.]
)

#figure(
  image("report_assets/all_methods_roc_comparison.png", width: 72%),
  caption: [ROC comparison across all evaluated methods. Supervised baselines dominate the ranking, while ConvAE + LOF is the best assignment-aligned detector.]
)

= Per-Machine Diagnostic and ID06 Ablation

The baseline results were also inspected by machine ID after the thresholds and final-test predictions had already been fixed. This diagnostic does not change the reported metrics; it only decomposes the errors by source machine. For ConvAE + LOF, ID `06` accounts for a large fraction of the normal false positives: 24 out of 47 normal ID `06` files are classified as anomalous, corresponding to a false-positive rate of 51.1%. The corresponding false-positive rates for IDs `00`, `02`, and `04` are 14.5%, 10.2%, and 16.3%. The same pattern is not present for reconstruction error, where normal ID `06` has the lowest false-positive rate among the four machine IDs. This suggests that the main ID `06` issue is the latent-space density model rather than reconstruction quality alone.

To test whether the absence of ID `06` from the ConvAE training set contributes to this behavior, a second diagnostic protocol was run. Protocol 2 adds a deterministic subset of threshold-normal ID `06` files to the normal-only ConvAE training and validation data, then refits the ConvAE-based unsupervised detectors. The final test split is unchanged, so the comparison uses the same 200 normal and 320 anomalous final-test files.

#figure(
  text(size: 7.8pt)[
    #table(
      columns: (2.05fr, 0.82fr, 0.78fr, 0.78fr, 0.78fr),
      align: (left, right, right, right, right),
      inset: 4pt,
      table.header([Protocol], [Bal Acc], [AUC-ROC], [ID06 FPR], [ID06 FNR]),
      [Baseline ConvAE + LOF], [0.733], [0.788], [51.1%], [31.9%],
      [Protocol 2 ConvAE + LOF], [0.737], [0.810], [40.4%], [34.8%],
    )
  ],
  caption: [Focused comparison for the ID06-normal-augmented ablation. The final test set is identical in both rows.]
)

The ablation supports the unseen-normal-machine hypothesis only partially. Adding ID `06` normal files reduces the ID `06` false-positive rate by about 10.6 percentage points and slightly improves overall LOF balanced accuracy and AUC-ROC. However, the ID `06` false-negative rate increases from 31.9% to 34.8%, and the ID `06` false-positive rate remains substantially higher than for the other machine IDs. The remaining error therefore indicates overlap in the latent representation, not only missing ID `06` coverage.

= Critical Discussion

The most important negative result is that reconstruction error does not separate normal and anomalous files well. The available checks do not indicate a preprocessing or training error: the ConvAE trains stably, the preprocessing is consistent, and the checkpoint matches the current spectrogram shape. The limitation is methodological. MSE on normalized log-mel images is a coarse anomaly score. A pump fault may alter local texture, harmonic stability, periodicity, or temporal consistency without producing a large average pixelwise reconstruction error. The decoder can reconstruct many anomalous spectrograms well enough that their MSE remains close to normal files.

The latent space is more useful than reconstruction error, but still imperfect. The t-SNE visualization shows partial structure rather than a clean class split. LOF benefits from this because it uses local density, while Isolation Forest and One-Class SVM struggle with the global boundary.

#figure(
  image("report_assets/tsne_embedding_visualization.png", width: 72%),
  caption: [Qualitative t-SNE view of ConvAE latent vectors on the test set. Normal and anomaly points are partially separated but still overlap, matching the moderate unsupervised metrics.]
)

Machine identity also contributes to the difficulty, but the ablation shows that this is not the whole explanation. Protocol 2 improves ConvAE + LOF modestly, yet it does not improve all ConvAE-based methods. Reconstruction error becomes more sensitive and produces many more false positives, Isolation Forest decreases in balanced accuracy, and One-Class SVM remains essentially unchanged in balanced accuracy. For this reason, Protocol 2 is best interpreted as an explanatory ablation rather than a replacement for the baseline protocol.

Per-machine threshold calibration was not adopted. Although it could reduce some ID-specific false positives, it would add a second calibration layer with relatively few files per machine ID. In this setting, such thresholds would risk fitting machine-specific validation artifacts rather than learning a general anomaly criterion. The report therefore keeps one global threshold per method and uses the per-machine analysis only to interpret the errors.

The supervised baselines should therefore be interpreted carefully. Their high AUC values show that the audio contains discriminative signal, so the weak reconstruction-error result should not be attributed to a complete absence of class information. However, they solve an easier problem because anomaly labels are available during training. In addition, their threshold tuning is somewhat optimistic because supervised training and threshold selection both use the threshold split, although the final test set remains held out. For this reason, they are explanatory baselines rather than the headline assignment result.

= Conclusion

The project satisfies the assignment objective by building a ConvAE latent feature space for water-pump audio and evaluating multiple anomaly-detection mechanisms on top of it. The best assignment-aligned method is ConvAE + LOF, with balanced accuracy 0.733 and AUC-ROC 0.788 on the held-out file-level test set. Reconstruction error alone is insufficient for this dataset, and the low separation is consistent with subtle pump-audio anomalies, overlap between normal and faulty acoustic patterns, and unseen normal machine variation.

Supervised baselines perform better, especially SVM on MFCC statistics, which indicates that discriminative information exists. However, those baselines use anomaly labels and therefore should be framed as diagnostic comparisons rather than replacements for the unsupervised ConvAE result.

The per-machine diagnostic and ID06-normal ablation indicate that unseen normal machine variation is one contributor to the error pattern, but not a complete explanation. Future improvements should therefore focus on richer audio representations, temporal aggregation beyond simple means, feature learning methods that separate machine identity from fault information, and alternative reconstruction or contrastive objectives. Per-machine thresholding would only be appropriate in a larger evaluation setting with enough calibration data per machine ID to control overfitting.
