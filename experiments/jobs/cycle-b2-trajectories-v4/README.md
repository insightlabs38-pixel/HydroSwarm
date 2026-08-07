# cycle-b2-trajectories-v4 regeneration

Regenerates the Strategist trajectory corpus using the corrected canonical
exposure-aware plan verification path (important-issues.txt fix, commit
96d945f). Preserves data/learning-v2/cycle-b2-trajectories-v3 untouched
(important-issues.txt restriction: "existing v3/v4 result artifacts
immutable").

Resumable: scripts/generate_trajectory_corpus.py skips any scenario_id
already present in the split's .jsonl shard on restart.
