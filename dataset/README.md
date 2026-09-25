# Dataset

600K~1M samples live here as parquet shards:

```
dataset/
  train-*.parquet
  val-*.parquet
  test-*.parquet
  soft_labels.parquet   # Teacher KD targets (Phase 2)
```

Row schema defined in `data_pipeline/__init__.py:Sample`.
