# Data directory

Generated and imported data are intentionally excluded from source control. Use `data/raw`
for immutable licensed inputs, `data/interim` for normalized networks, `data/processed` for
split scenario shards, and `data/frozen` only for small deterministic regression fixtures.
Every generated dataset must include a manifest with source licenses, seed, generator
version, scenario hashes, and network-disjoint split assignments.
