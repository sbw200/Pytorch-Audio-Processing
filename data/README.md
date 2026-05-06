# Data

This project is configured for UrbanSound8K.

The active dataset is expected at the project root as:

```text
archive (4)/
  UrbanSound8K.csv
  fold1/
  fold2/
  fold3/
  fold4/
  fold5/
  fold6/
  fold7/
  fold8/
  fold9/
  fold10/
```

The config points to:

```yaml
data:
  dataset_root: archive (4)
  metadata_csv: archive (4)/UrbanSound8K.csv
```

The `data/raw/` and `data/processed/` folders are available for optional local experiments, but the implemented dataset loader reads directly from the UrbanSound8K archive path above.

Do not commit large audio datasets or generated feature dumps unless explicitly needed.
