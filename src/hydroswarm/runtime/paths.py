"""Single source of truth for resolving the frozen HydroCore-v4 release
bundle directory.

Before this module existed, `hydroswarm.api.app` computed the bundle
location from `Path(__file__).resolve().parents[3]` (the source-tree
layout) while `hydroswarm.cli.run_self_test` independently computed it from
`Path.cwd()`. The two calculations happen to agree for an editable/source
checkout run from the repository root, but they diverge -- silently, with
no error -- for:

* a non-editable ("regular") `pip install .`, where the installed package
  lives under `site-packages/hydroswarm/...` and `parents[3]` from there is
  not the repository root at all;
* any invocation whose current working directory is not the repository
  root (a systemd unit, a different launch script, a container `WORKDIR`
  that does not match the source layout).

A production server and its own self-test silently checking two different
directories is exactly the "appears healthy while actually degraded" failure
submission-readiness explicitly rules out. `resolve_v4_bundle_dir` is the one
function both entry points call, so they can only ever agree or fail
identically.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Explicit override. Highest priority. This is what the container image
#: sets (to the bundle baked into the image) and what any other packaged or
#: non-standard deployment should set rather than relying on source-tree-
#: relative inference.
V4_BUNDLE_DIR_ENV_VAR = "HYDROSWARM_V4_BUNDLE_DIR"

#: Explicit override for the writable runtime data directory (database,
#: signature cache, imported networks). Matches the variable name already
#: declared -- but, until this module, never actually read -- by the
#: Dockerfile and docker-compose.yml.
DATA_DIR_ENV_VAR = "HYDROSWARM_DATA_DIR"

_BUNDLE_RELATIVE_PATH = ("models", "hydrocore-v4-release")


def _source_tree_project_root() -> Path:
    """The repository root, inferred from this file's own location.

    Only valid for an editable/source checkout (`src/hydroswarm/runtime/
    paths.py` -> repository root is 3 parents up). For a non-editable
    install this still returns *a* directory, but it will not contain
    `models/`; callers must not treat this as authoritative without
    checking the candidate path actually exists.
    """
    return Path(__file__).resolve().parents[3]


def resolve_v4_bundle_dir(project_root: str | Path | None = None) -> Path:
    """Resolve the frozen HydroCore-v4 release bundle directory.

    Resolution priority:

    1. The `HYDROSWARM_V4_BUNDLE_DIR` environment variable, if set --
       the explicit override used by the container image and any other
       packaged deployment.
    2. `models/hydrocore-v4-release` relative to `project_root` (or, if not
       given, this module's inferred source-tree root) -- the packaged
       application default for an editable/source checkout.
    3. `models/hydrocore-v4-release` relative to the current working
       directory, if that exists and (2) does not -- a development
       fallback for a non-editable install invoked from within a checked-
       out repository.

    This never raises for a missing bundle: callers (`V4PipelineFactory`)
    already fail closed on a missing/invalid directory, and returning a
    deterministic "best guess" path here keeps that fail-closed error
    message meaningful instead of substituting an ambiguous one.
    """
    override = os.environ.get(V4_BUNDLE_DIR_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()

    root = Path(project_root).resolve() if project_root is not None else _source_tree_project_root()
    source_tree_candidate = root.joinpath(*_BUNDLE_RELATIVE_PATH)
    if source_tree_candidate.is_dir():
        return source_tree_candidate.resolve()

    cwd_candidate = Path.cwd().joinpath(*_BUNDLE_RELATIVE_PATH)
    if cwd_candidate.is_dir():
        return cwd_candidate.resolve()

    return source_tree_candidate.resolve()


def resolve_data_dir(project_root: str | Path | None = None) -> Path:
    """Resolve the writable runtime data directory (database, signature
    cache, imported-network storage).

    Priority: `HYDROSWARM_DATA_DIR` if set (the container image's writable
    volume mount), else `<project_root>/data/generated` for local/dev runs
    -- matching this project's pre-existing default database location
    (`hydroswarm.storage.database.default_database_path`).
    """
    configured = os.environ.get(DATA_DIR_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    root = Path(project_root).resolve() if project_root is not None else _source_tree_project_root()
    return (root / "data" / "generated").resolve()
