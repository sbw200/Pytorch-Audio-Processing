# Best Practices

## Environment

Use `uv` for environment management and keep dependencies in `pyproject.toml`.

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -e .
```

Do not add `requirements.txt` unless the project decision changes.

## Data

Keep the extracted UrbanSound8K archive in the configured location:

```text
archive (4)/UrbanSound8K.csv
archive (4)/fold1 ... archive (4)/fold10
```

Use the official UrbanSound8K folds. The validation fold is controlled by:

```yaml
data:
  validation_fold: 1
```

Avoid committing large audio archives, generated datasets, or copied raw data.

## Preprocessing

Training, evaluation, and inference should use the same preprocessing path:

```text
load waveform -> mono -> resample -> normalize -> trim/pad -> Mel spectrogram -> optional dB
```

The current expected model input is:

```text
[batch_size, 1, 64, 126]
```

This comes from `sample_rate: 16000`, `duration_seconds: 4.0`, `n_mels: 64`, and `hop_length: 512`.

## Training

Run training from the project root:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\config.yaml
```

Use smoke-test limits only to verify wiring:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\config.yaml --epochs 1 --max-train-batches 1 --max-val-batches 1
```

Full training should produce:

```text
outputs/checkpoints/best_model.pt
outputs/metrics/training_history.json
outputs/plots/training_curves.png
```

The best checkpoint is selected by validation macro F1.

## Evaluation

Evaluate the saved checkpoint with:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --config configs\config.yaml --checkpoint outputs\checkpoints\best_model.pt
```

Evaluation should produce:

```text
outputs/metrics/evaluation_metrics.json
outputs/plots/confusion_matrix.png
```

Track macro F1 along with accuracy because UrbanSound8K classes are not perfectly balanced.

## Inference

Use the same config and checkpoint as training:

```powershell
.\.venv\Scripts\python.exe scripts\simulate_edge_inference.py --config configs\config.yaml --checkpoint outputs\checkpoints\best_model.pt --audio_path "path\to\file.wav"
```

The reported inference time is the model forward pass only. It does not include file loading or preprocessing.

## Outputs

Treat files under `outputs/` as generated artifacts. They can be overwritten by smoke tests or full training runs.

If comparing experiments, copy or rename the relevant checkpoint, metrics JSON, and plot files before starting another run.
