# Security, Safety, and Product Boundary

HydroSwarm is local decision support. It does not connect to SCADA, operate valves or
pumps, fetch remote URLs, or execute commands supplied through incident data. The default
server address is `127.0.0.1`; widening the bind address requires an explicit deployment
security review.

Every operational payload is schema-validated. Every proposed plan must be prescreened and
simulated. A verified result remains pending until an operator records approval. Rejected
actions, simulator versions, state hashes, and approvals are stored in an append-only,
hash-chained SQLite ledger.

Network imports are restricted to `.inp` files: the importer enforces an `.inp`-only
filename allowlist with no embedded paths, a 5 MiB size limit, UTF-8 text with binary/NUL
content rejected, a line-count limit, and required hydraulic sections. External `[FILES]`
references inside an uploaded `.inp` are rejected rather than followed. Accepted content is
hashed and stored under path-contained, sanitized names, then parsed and hydraulically
validated before use. Archive formats and other arbitrary file types are not accepted.

Known limitations include synthetic-only scenarios, simplified contaminant transport in
the generator, no chemistry identification, no production safety guarantee, no causal
interpretation of sensitivity, and no demonstrated universal cross-network generalization.
Operators must follow established utility procedures and qualified engineering judgment.

The release audit exports a hash-locked dependency report, a CycloneDX SBOM, and a
deterministic credential-pattern scan under `reports/results/`. The credential scan covers
tracked and release-bound untracked text files and fails on common cloud, hosted-model,
GitHub-token, and private-key signatures.
