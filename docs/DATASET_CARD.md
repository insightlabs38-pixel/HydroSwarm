# Synthetic Scenario Dataset Card

HydroSwarm's generator creates deterministic network and sensor scenarios for engineering
tests. Networks include elevations, a reservoir, tank capacity, loops, and diurnal demand.
Incident streams include a transient source pulse and topology-weighted transport delay.

Imperfect-condition controls include Gaussian noise, signed drift, missingness, observation
latency, and flow-reversal masks. Each output aligns truth, timestamps, values, and quality
masks. Seeds are explicit and replayable.

The committed learning-v1 evidence contains 1,320 deterministic WNTR incidents: 800 train,
160 validation, 160 calibration, and 200 test. Each of five curriculum stages contains 264
incidents; each of four source nodes contains 330. The corpus spans four training hydraulic
regimes and one withheld test regime, records 100% replay validation and zero quarantines,
and occupies 23,289,090 bytes before excluding regenerable NPZ/Parquet binaries from Git.
Canonical tensor manifests use the exact production feature builder and schema hash.

This corpus is not a substitute for utility datasets. Its five named regimes share one
topology, so the held-out result measures hydraulic-parameter shift rather than topology
transfer. Future evaluation must add independently structured networks such as C-Town,
prevent seed-family, source-ID, future-observation, and simulator-label leakage, and
quarantine nonfinite or nondeterministic scenarios.

Recommended storage for expanded experiments is NumPy/Zarr for arrays, Parquet for tables,
JSONL for manifests/events, `.inp` for networks, and safetensors for checkpoints. Do not
commit generated incident data or sensitive utility network files.
