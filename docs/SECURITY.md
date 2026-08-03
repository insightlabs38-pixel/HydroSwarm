# Security, Safety, and Product Boundary

HydroSwarm is local decision support. It does not connect to SCADA, operate valves or
pumps, fetch remote URLs, or execute commands supplied through incident data. The default
server address is `127.0.0.1`; widening the bind address requires an explicit deployment
security review.

Every operational payload is schema-validated. Every proposed plan must be prescreened and
simulated. A verified result remains pending until an operator records approval. Rejected
actions, simulator versions, state hashes, and approvals are stored in an append-only,
hash-chained SQLite ledger.

Before accepting arbitrary files, enforce `.inp` allowlisting, file and expanded-size
limits, sanitized names, path containment, content parsing, hashes, controlled storage,
safe CORS, and report path redaction. Do not accept archive formats unless hardened archive
validation is implemented.

Known limitations include synthetic-only scenarios, simplified contaminant transport in
the generator, no chemistry identification, no production safety guarantee, no causal
interpretation of sensitivity, and no demonstrated universal cross-network generalization.
Operators must follow established utility procedures and qualified engineering judgment.

