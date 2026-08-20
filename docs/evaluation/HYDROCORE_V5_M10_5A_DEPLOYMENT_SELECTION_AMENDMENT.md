# M10.5A deployment identity selection amendment

Frozen from `cecee4c8a8bd9bc3801d8038bfe9b858508bc8fd`, before any v5 bundle,
serving wiring, parity run, or M10.4 per-seed performance inspection.

The M10.5 blocker is resolved solely by the historical canonical seed order.
`scripts/hydrocore_v5/m9_4_common.py` declared `SEEDS = (20260814, 31874,
20260815)` in commit `f2e7857f00be6e33420439f44b6ededa0e6c396f` on 2026-08-16;
M9.6 and M10 re-export that tuple unchanged.  This predates M10.4.

The frozen rule is:

`FIRST_CANONICAL_SEED_IN_PREEXISTING_FROZEN_SEED_ORDER`

It chooses seed `20260814` and only its M9.6 ARM_B canonical
`FINAL_STEP_1350` export.  It is not a claim that this seed is better than
its canonical peers and must never consume M10.4 per-seed metrics.  All three
peers remain historical canonical checkpoints.

If the audit cannot prove that provenance, M10.5A blocks and M10.5 cannot
resume.  On pass, this amendment is committed independently before serving
implementation begins.
