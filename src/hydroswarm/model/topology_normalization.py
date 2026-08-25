"""Experimental, additive topology-relative feature augmentation.

Not wired into any production default, factory, dataset, or promoted
checkpoint -- see
docs/evaluation/experimental/TOPOLOGY_GENERALIZATION_EXPERIMENT_PLAN.md for
the hypothesis this exists to test. `HydroCore` (`hydroswarm.model.core`)
already accepts `node_feature_dim`/`edge_feature_dim` as plain constructor
arguments and never imports outside `hydroswarm.model`, so this module is
deliberately applied one layer up, as a batch post-processing step after
`hydroswarm.training.variable_collate.collate_variable_topology` (see
`augment_batch` below) in the experiment's own training/evaluation scripts
-- not inside `HydroCore.forward()` -- so zero production module (`core.py`,
`layers.py`, `encoders.py`, any factory, any dataset loader) is modified by
this experiment. A caller that wants the augmented representation builds
`HydroCore(node_feature_dim=augmented_width(19, NODE_TOPOLOGY_RELATIVE_COLUMNS), ...)`
and applies `augment_batch` to every collated batch; every existing
caller/checkpoint that does neither is completely unaffected.

`hydroswarm.preprocessing.schema` already tags every node/edge feature
`FeatureScope.ABSOLUTE` or `FeatureScope.TOPOLOGY_RELATIVE` (documentation
only, previously unused to change any actual normalization).
`node_features`/`edge_features` are otherwise normalized once, globally, by
a `NormalizationStats` fit on the train split only
(`hydroswarm.preprocessing.builder`) -- a scale that does not adapt to a
served network whose absolute size/terrain/demand differs from training.
`hydroswarm.model.encoders.GraphStructuralEncoder` already avoids exactly
this problem for its own 3 structural scalars via a per-example (per-graph)
max-abs rescale before its own projection. This module applies the same
per-graph rescale convention to the columns `schema.py` already marks
`TOPOLOGY_RELATIVE`, appending them as additional columns (never replacing
the original, globally-normalized ones) so the model can learn to use
either representation.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hydroswarm.preprocessing.schema import (
    EDGE_FEATURE_NAMES,
    EDGE_FEATURE_SEMANTICS,
    NODE_FEATURE_NAMES,
    NODE_FEATURE_SEMANTICS,
    FeatureScope,
)


def _topology_relative_columns(
    names: tuple[str, ...], semantics: dict[str, "object"]
) -> tuple[int, ...]:
    return tuple(
        index
        for index, name in enumerate(names)
        if semantics[name].scope == FeatureScope.TOPOLOGY_RELATIVE
    )


#: Indices into NODE_FEATURE_NAMES / EDGE_FEATURE_NAMES of every feature
#: schema.py already documents as topology-relative (scale depends on that
#: network's own size/terrain/demand), derived from the schema rather than
#: hand-duplicated so this module can never silently drift out of sync with
#: it.
NODE_TOPOLOGY_RELATIVE_COLUMNS: tuple[int, ...] = _topology_relative_columns(
    NODE_FEATURE_NAMES, NODE_FEATURE_SEMANTICS
)
EDGE_TOPOLOGY_RELATIVE_COLUMNS: tuple[int, ...] = _topology_relative_columns(
    EDGE_FEATURE_NAMES, EDGE_FEATURE_SEMANTICS
)


def augmented_width(base_width: int, columns: tuple[int, ...]) -> int:
    """Encoder input width after appending the relative-normalized copy of
    `columns`. Used to size `StaticFeatureEncoder`/the graph backbone's edge
    projection when `topology_relative_augmentation=True`."""

    return base_width + len(columns)


def augment_with_relative_scale(
    features: Tensor | None,
    mask: Tensor | None,
    columns: tuple[int, ...],
) -> Tensor | None:
    """Append a per-example (per-graph), mask-aware, max-abs-normalized copy
    of `columns` to `features`.

    `features`: `[batch, items, width]` (nodes or edges). `mask`:
    `[batch, items]`, True for valid items, or None (all valid). Padded/
    invalid items are excluded from the per-graph scale statistic and their
    appended columns are zeroed, matching how the rest of HydroCore's
    masked reductions behave. Returns `features` unchanged if it is None
    (e.g. no edges present in a batch) or `columns` is empty.
    """

    if features is None or not columns:
        return features
    if features.ndim != 3:
        raise ValueError("features must have shape [batch, items, width]")
    selected = torch.nan_to_num(features[..., columns].float(), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is None:
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
    else:
        valid = mask.bool()
        if valid.shape != features.shape[:2]:
            raise ValueError("mask must have shape [batch, items]")
    valid_column = valid.unsqueeze(-1)
    masked = torch.where(valid_column, selected, torch.zeros_like(selected))
    scale = masked.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    relative = torch.where(valid_column, selected / scale, torch.zeros_like(selected))
    return torch.cat((features, relative.to(features.dtype)), dim=-1)


def augment_batch(inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a shallow-copied batch dict with `node_features`/
    `edge_features` augmented by `augment_with_relative_scale`, using
    whatever `node_mask`/`edge_mask` the batch already carries. Intended to
    be composed with `hydroswarm.training.variable_collate.collate_variable_topology`,
    e.g. `collate_fn = lambda examples: augment_batch(collate_variable_topology(examples)[0]) | (collate_variable_topology(examples)[1],)`
    -- in practice the experiment's own collate wrapper calls
    `collate_variable_topology` once and passes its `inputs` dict here
    directly (see `scripts/hydrocore_v5_experimental/topology_generalization/data.py`).
    Duck-typed on plain `dict[str, Tensor]` so this stays free of any
    import from `hydroswarm.training`, preserving `hydroswarm.model`'s
    existing layering.
    """

    augmented = dict(inputs)
    augmented["node_features"] = augment_with_relative_scale(
        inputs.get("node_features"), inputs.get("node_mask"), NODE_TOPOLOGY_RELATIVE_COLUMNS
    )
    augmented["edge_features"] = augment_with_relative_scale(
        inputs.get("edge_features"), inputs.get("edge_mask"), EDGE_TOPOLOGY_RELATIVE_COLUMNS
    )
    return augmented
