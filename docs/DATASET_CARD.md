# Synthetic Scenario Dataset Card

HydroSwarm's generator creates deterministic network and sensor scenarios for engineering
tests. Networks include elevations, a reservoir, tank capacity, loops, and diurnal demand.
Incident streams include a transient source pulse and topology-weighted transport delay.

Imperfect-condition controls include Gaussian noise, signed drift, missingness, observation
latency, and flow-reversal masks. Each output aligns truth, timestamps, values, and quality
masks. Seeds are explicit and replayable.

This compact generator is not a substitute for utility datasets. Future evaluation should
use network-family-separated splits, with hashes and manifests generated before simulation;
prevent seed-family, source-ID, future-observation, and simulator-label leakage; quarantine
nonfinite or nondeterministic scenarios; and hold out entire networks such as C-Town.

Recommended storage for expanded experiments is NumPy/Zarr for arrays, Parquet for tables,
JSONL for manifests/events, `.inp` for networks, and safetensors for checkpoints. Do not
commit generated incident data or sensitive utility network files.

