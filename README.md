# Audio Anomaly Detection on Water Pumps

University project for anomaly detection on water-pump audio. The canonical pipeline is:

1. Convert audio windows to fixed log-mel spectrograms.
2. Train a convolutional autoencoder on normal audio only.
3. Use the ConvAE latent representation for anomaly detection with classical methods.

The main experiment artifact is `anomaly_detection.ipynb`. The written report is generated from `report.typ` and compiled to `report.pdf`.

## Dataset Layout

Place raw `.wav` files under:

```text
data/raw/
  train-normal/
  test-normal/
  anomaly/
```

Splits are made at file level with seed `42`:

- `train-normal`: ConvAE train/validation.
- `test-normal`: threshold tuning and final normal test.
- `anomaly`: threshold tuning and final anomaly test.

Each 10-second file is processed into 2-second windows with 1-second hop, giving 9 windows per file. Metadata includes both the basename and a stable `source_id` such as `train-normal/example.wav` so equal basenames in different folders are not merged.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`panns-inference` is optional for the PANNs sanity baseline. The assignment-focused ConvAE and classical latent-space methods do not require running PANNs.

## Running The Notebook

Open and run:

```bash
jupyter notebook anomaly_detection.ipynb
```

The notebook:

- preprocesses or loads cached spectrograms from `data/processed/`;
- trains or loads `checkpoints/conv_ae_best.pt`;
- extracts ConvAE latent vectors;
- evaluates reconstruction error, Isolation Forest, LOF, One-Class SVM, supervised SVM, and XGBoost;
- reports file-level metrics using balanced-accuracy threshold tuning.

If cached metadata lacks `source_id`, the notebook migrates the metadata in place when the original split file list is available; it reprocesses only if migration is impossible.

Each notebook run also creates a timestamped artifact directory under `outputs/notebook_runs/`. Figures are saved as PNG files, printable comparison tables are saved as TXT/CSV files, and raw metric dictionaries are saved as JSON. The `outputs/` directory is ignored by Git because these files are generated run artifacts.

## Building The Report

The report source is written in Typst and uses stable copies of selected notebook figures under `report_assets/`:

```bash
typst compile report.typ report.pdf
```

## Optional PANNs Baseline

PANNs embeddings are treated as an optional upper-bound/sanity baseline and are disabled by default in the notebook. Set `RUN_PANNS_BASELINE = True` in `anomaly_detection.ipynb` to enable cached PANNs evaluation or extraction. To extract new embeddings, also set:

```bash
export PANNS_CHECKPOINT_PATH=/absolute/path/to/Cnn14_mAP=0.431.pth
```

The notebook prefers cached PANNs embeddings when present. New caches are saved as `*_embeddings.npy`; old `*_spectrograms.npy` cache names are still supported for compatibility.

## Tests

Synthetic tests cover the reusable code paths and do not require the raw audio dataset:

```bash
python -m pytest
```

The tests check metadata identity, file-level score aggregation, score-derived threshold search, grouped supervised CV, and basic model-scoring helper behavior.
