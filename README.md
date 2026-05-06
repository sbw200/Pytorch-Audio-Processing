# PyTorch Audio Processing

UrbanSound8K audio classification project using PyTorch, torchaudio transforms, and a lightweight CNN over Mel spectrograms.

## Setup

Create and activate a uv virtual environment:

```powershell
uv venv
.\.venv\Scripts\Activate.ps1
```

Install the project in editable mode:

```powershell
uv pip install -e .
```

Dependencies are defined in `pyproject.toml`. This project intentionally does not use `requirements.txt`.

## Data

The current config expects the extracted UrbanSound8K dataset here:

```text
archive (4)/
  UrbanSound8K.csv
  fold1/
  fold2/
  ...
  fold10/
```

Configured paths are in `configs/config.yaml`:

```yaml
data:
  dataset_root: archive (4)
  metadata_csv: archive (4)/UrbanSound8K.csv
  validation_fold: 1
```

Classes:

```text
0 air_conditioner
1 car_horn
2 children_playing
3 dog_bark
4 drilling
5 engine_idling
6 gun_shot
7 jackhammer
8 siren
9 street_music
```

## Quick Checks

Verify the dataset loader and preprocessing:

```powershell
.\.venv\Scripts\python.exe scripts\check_dataset_loader.py
```

Expected shape with the current config:

```text
(1, 64, 126)
```

That means one mono Mel spectrogram with 64 Mel bins and 126 time frames.

## Training

Train the CNN:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\config.yaml
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\config.yaml --epochs 1 --max-train-batches 1 --max-val-batches 1
```

Training writes:

```text
outputs/checkpoints/best_model.pt
outputs/metrics/training_history.json
outputs/plots/training_curves.png
```

## Evaluation

Evaluate a checkpoint on the validation fold:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --config configs\config.yaml --checkpoint outputs\checkpoints\best_model.pt
```

Evaluation writes:

```text
outputs/metrics/evaluation_metrics.json
outputs/plots/confusion_matrix.png
```

## Single-File Inference

Run one WAV file through the trained model:

```powershell
.\.venv\Scripts\python.exe scripts\simulate_edge_inference.py --config configs\config.yaml --checkpoint outputs\checkpoints\best_model.pt --audio_path "archive (4)\fold1\101415-3-0-2.wav"
```

The script prints:

```text
predicted_class: class name with the highest probability
confidence: softmax confidence for that prediction
inference_time_ms: model forward-pass time only
```

## Project Layout

```text
configs/              Configuration files
data/                 Data notes and optional local data folders
docs/                 Project documentation
outputs/checkpoints/  Saved model checkpoints
outputs/metrics/      Training and evaluation JSON files
outputs/plots/        Training curves and confusion matrix plots
scripts/              Utility scripts and inference simulation
src/                  Dataset, preprocessing, model, training, and evaluation code
```
