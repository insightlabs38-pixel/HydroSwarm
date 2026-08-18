# HydroCore v5 M10.5 serving-path freeze protocol

This frozen completion protocol consumes only M10.5A's selected seed
`20260814` and M10.5B's immutable calibration artifact. It creates an
immutable v5 bundle, changes the normal default application to that bundle,
and evaluates a preregistered small development-only parity/safety smoke set.
No checkpoint or calibration selection, retraining, threshold adjustment,
feature-semantics change, or locked evaluation access is authorized.

The v5 runtime must validate its own model, calibration, schema, fusion,
topology, and manifest identities; failure leaves only deterministic
classical-safe behavior. It must never substitute the v4 bundle. Learned OOD,
Scout, and Strategist roles remain suppressed by `trained_tasks={"sentinel"}`.
The serving allowlist is exactly `event_cause`, `event_presence`,
`evidence_sufficiency`, `relative_strength`, and `source_node`; `next_step`
is deliberately suppressed because M9.6 did not supervise it. M10.4's
unobserved-age serving semantics are retained unchanged and recorded as a
known train/serve deviation, not repaired here.
