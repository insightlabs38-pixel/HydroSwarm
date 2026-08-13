# LIVE robustness harness correction log

The first complete, uncommitted campaign execution produced 264 deterministic
rows. It is superseded before interpretation by the same 264 run IDs after
the two harness-only corrections below. No production code, model artifact,
threshold, calibration artifact, feature schema, simulator policy, or locked
test state changed.

1. Rows `68ccef51de56c041`, `0139c2a9bf1e0704`, `c96299b2a6b55b5d`,
   `92383080870e1cb5`, `9190eb298ec52941`, `6711c18d52eeb486`,
   `0ff25b18db7bc927`, `0faafdbd154fb090`, and `47c17ce6333ac885` returned
   the real API's `ANALYZE_409` for no usable evidence. The original harness
   labelled these `HARNESS_ERROR`; this was inaccurate. They are rerun and
   labelled `ABSTAINED`, retaining null unavailable metrics and the HTTP
   class. This is expected fail-closed authority behavior, not a product
   defect.
2. The original loop-grid lifecycle rows verified all requested plans before
   attempting approval. That made a successful approval transition
   unobservable in the measured trajectory. The corrected lifecycle slice
   uses one plan for the predeclared controlled approval or stale-verification
   trajectory: verify then approve, or verify then mutate evidence then prove
   approval rejection. The ordinary measurement trajectory still asks for two
   plan candidates.

The correction also continues to stop rather than insert duplicate synthetic
evidence when the real sampler recommends an already-observed node. That
behavior is retained as `ROB-LIVE-01`; it is not a harness error.

3. The initial topology-familiarity rows named `branched-loop` as
   `development_unseen`. A post-run structural-hash check established that
   its topology hash is `0b1817cd…`, one of the calibration artifact's
   validated hashes. Those 24 rows are therefore not evidence about
   unfamiliar topology. The same deterministic `unseen-*` run IDs are
   rerun against committed `coastal-branch.inp` (structural hash
   `d8725933…`, absent from the validated hashes) before results are
   interpreted. This corrects a fixture label and population selection; it
   does not change an experiment based on whether its measured result was
   favorable.
