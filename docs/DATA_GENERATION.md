# Data generation and governance

HydroSwarm trains and evaluates on WNTR-generated trajectories, not arbitrary feature
tables. The generator randomizes incident source, timing, duration/rate profile, demands,
tank state, roughness, pump/valve outages, sensor placement, noise, bias, drift, freezing,
outage, jitter, unit mismatch, and flow reversal. Every label is tied to a simulator run.

## Required workflow

1. Place immutable, licensed network files in `data/raw/<source>` and record source URL,
   retrieval date, checksum, and license. Never commit a restricted dataset.
2. Normalize locally to `data/interim` and validate hydraulics.
3. Assign network-disjoint train/calibration/validation/test splits **before** scenario
   simulation. A network or derived version may appear in only one split.
4. Generate seeded shards to `data/processed`, with JSONL metadata, compressed NPZ tensors,
   and Parquet indexes.
5. Run finite-value, unit, schema, replay, leakage, and checksum validation.
6. Freeze only compact golden fixtures under `data/frozen`.

```powershell
python scripts/generate_dataset.py --config configs/training.yaml --output data/processed
python scripts/build_signatures.py --data data/processed --output models/signatures
python scripts/calibrate.py --data data/processed --output models/calibration
python scripts/train.py --config configs/training.yaml
python scripts/evaluate.py --config configs/evaluation.yaml
```

The exact command interfaces are self-documenting with `--help`. Generated manifests
must include generator/git versions, network and scenario hashes, seeds, physical units,
time grid, feature schema, corruption settings, split assignments, and provenance.

## External networks

Useful public candidates include the University of Kentucky Water Distribution System
Research Database and public EPA catalog entries. Their files are not redistributed here.
Users are responsible for verifying current terms and attribution, preserving original
checksums, and documenting transformations. Synthetic reference networks remain the
reproducible fallback when external licensing or availability is uncertain.

## Leakage controls

The validator rejects shared network hashes, derived topology hashes, or scenario IDs
across partitions. Calibration is separate from model training and final test data.
Hard-negative mining may append training examples only; test and frozen fixtures are
immutable. Dataset aggregation records its parent model and never retroactively changes a
reported evaluation split.
