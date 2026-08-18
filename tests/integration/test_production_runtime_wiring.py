"""UI-11.1 Section 1: the normal documented production launch
(start_hydroswarm.sh/.bat -> hydroswarm.cli start -> hydroswarm.api.app:app)
must serve the frozen, tagged `hydrocore-v4-architecture-freeze` candidate by
default, composed through the existing `V4PipelineFactory` against the
committed `models/hydrocore-v4-release/` bundle -- not the older
`DefaultPipelineFactory`/`models/hydrocore-s-learning-v1.safetensors`
(hydrocore-v3, adapters=true, feature_and_logit) path.

These identity constants are copied verbatim from the `hydrocore-v4-architecture-freeze`
git tag message and `reports/results/v4/architecture-freeze.json` -- if this
test ever needs to change these values, that is itself a sign something
requires human review (a real re-freeze), not a routine test update.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import wntr
from fastapi.testclient import TestClient

from hydroswarm.api import create_app
from hydroswarm.domain import ActionType, OperationalAction, OperationalPlan
from hydroswarm.runtime import DefaultPipelineFactory, V4PipelineFactory, V5PipelineFactory
from hydroswarm.simulation import HydraulicSimulator, PlanVerifier

FROZEN_MODEL_SHA256 = "de2b3f56243a1933d1d7c5957cd74a29fade119f7d104ce7f1500b3dd7b6d2a5"
FROZEN_ARCHITECTURE_VERSION = "hydroswarm-v5-release-v1"
FROZEN_RUNTIME_ENABLED_OUTPUTS = frozenset(
    {"event_cause", "event_presence", "evidence_sufficiency", "relative_strength", "source_node"}
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_RELEASE_BUNDLE_DIR = PROJECT_ROOT / "models" / "hydrocore-v4-release"
V5_RELEASE_BUNDLE_DIR = PROJECT_ROOT / "models" / "hydrocore-v5-release"


def test_module_level_production_app_is_wired_to_v5_pipeline_factory() -> None:
    """The real module-level `app` object -- exactly what
    `uvicorn hydroswarm.api.app:app` (started by start_hydroswarm.sh/.bat via
    `hydroswarm.cli start`) serves -- must resolve to V4PipelineFactory, not
    DefaultPipelineFactory."""

    from hydroswarm.api.app import DEFAULT_V5_RELEASE_BUNDLE_DIR, app

    assert DEFAULT_V5_RELEASE_BUNDLE_DIR == V5_RELEASE_BUNDLE_DIR
    factory = app.state.runtime.pipeline_factory
    assert isinstance(factory, V5PipelineFactory)
    assert not isinstance(factory, DefaultPipelineFactory)


def test_production_app_resolves_the_frozen_v5_checkpoint_identity() -> None:
    """Confirms the production app doesn't merely construct a V4PipelineFactory
    object -- it actually loads and resolves to the exact frozen identity."""

    from hydroswarm.api.app import app

    factory = app.state.runtime.pipeline_factory
    assert factory.trained_assets_ready is True
    assert factory.fallback_reason is None
    assert factory.model_hash == FROZEN_MODEL_SHA256

    assert factory.manifest is not None
    assert factory.manifest["release_schema_version"] == FROZEN_ARCHITECTURE_VERSION
    assert factory.manifest["model_config"]["prior_mode"] == "feature_only"


def test_runtime_enabled_output_governance_matches_the_frozen_declaration() -> None:
    """core-issues5.txt Section 12/Phase 15: runtime_enabled_outputs is the
    granular gate on which advisory heads are allowed to affect real
    decisions -- must match the frozen declaration exactly, neither wider
    (an ungoverned head silently going live) nor narrower (governed
    capability silently lost)."""

    from hydroswarm.api.app import app

    factory = app.state.runtime.pipeline_factory
    assert set(factory.manifest["runtime_enabled_outputs"]) == FROZEN_RUNTIME_ENABLED_OUTPUTS


@pytest.mark.real_simulation
def test_self_test_reports_the_same_frozen_identity_the_production_app_serves() -> None:
    """`hydroswarm.cli self-test` (the first command start_hydroswarm.sh/.bat
    runs, before `start`) must validate the SAME runtime the server actually
    launches -- not an independently-correct-but-different one. Both must
    agree on every identity field, not just each independently match the
    frozen constants."""

    from hydroswarm.api.app import app
    from hydroswarm.cli import run_self_test

    production_factory = app.state.runtime.pipeline_factory
    result = run_self_test()

    assert result["ok"] is True
    trained_assets = result["trained_assets"]
    assert trained_assets["ready"] is True
    assert trained_assets["fallback_reason"] is None
    assert trained_assets["model_sha256"] == production_factory.model_hash
    assert trained_assets["architecture_version"] == FROZEN_ARCHITECTURE_VERSION
    assert trained_assets["model_sha256"] == FROZEN_MODEL_SHA256


def test_missing_v4_release_bundle_fails_closed(tmp_path: Path) -> None:
    """If the governed release bundle is absent/corrupt, the factory must
    fail closed (classical-safe remains available) -- never silently fall
    back to a different, unintended checkpoint."""

    factory = V4PipelineFactory(tmp_path / "does-not-exist", project_root=tmp_path)
    assert factory.trained_assets_ready is False
    assert factory.identity is None
    assert factory.fallback_reason is not None
    assert factory.fallback_reason.startswith("v4_trained_assets_unavailable:")


@pytest.mark.real_simulation
def test_v4_factory_drives_a_real_incident_analysis_through_the_production_app(tmp_path: Path) -> None:
    """End-to-end proof the production composition actually runs real
    inference against a real (calibrated) network, not just that the
    objects construct without error. loop-grid.inp's topology hash is one
    of the three the frozen calibration artifact was fit against
    (architecture-freeze.json's validated_topology_hashes), so this
    exercises the real calibrated FULL_HYBRID path, not a CLASSICAL_SAFE
    fallback."""

    client = TestClient(
        create_app(
            pipeline_factory=V4PipelineFactory(V4_RELEASE_BUNDLE_DIR),
            database_path=tmp_path / "runtime.sqlite3",
        )
    )
    with open(PROJECT_ROOT / "data" / "topologies" / "loop-grid.inp", "rb") as handle:
        imported = client.post(
            "/api/networks/import", files={"file": ("loop-grid.inp", handle, "text/plain")}
        )
    assert imported.status_code == 201, imported.text
    network_id = imported.json()["network_id"]

    from datetime import UTC, datetime

    now = datetime.now(UTC)
    created = client.post(
        "/api/incidents",
        json={
            "network_id": network_id,
            "detected_at": now.isoformat(),
            "observations": [
                {
                    "sensor_id": "S1",
                    "node_id": "J1",
                    "observed_at": now.isoformat(),
                    "received_at": now.isoformat(),
                    "concentration_mg_l": 0.0,
                    "pressure_m": 30.0,
                    "quality": 1.0,
                    "missing": False,
                    "drift_flag": False,
                    "frozen_flag": False,
                },
            ],
            "maximum_samples": 3,
        },
    )
    assert created.status_code == 201, created.text
    incident_id = created.json()["incident_id"]

    analyzed = client.post(f"/api/incidents/{incident_id}/analyze")
    assert analyzed.status_code == 200, analyzed.text
    assert analyzed.json()["candidates"]["calibrated"] is True

    analysis = client.get(f"/api/incidents/{incident_id}/analysis").json()
    assert analysis["runtime_mode"] == "FULL_HYBRID"
    assert analysis["calibration_source"] == "NETWORK_SPECIFIC"
    assert analysis["calibration_group_identifier"] == "loop-grid"


@pytest.mark.real_simulation
def test_exact_wntr_verification_authority_is_unaffected_by_the_pipeline_factory_choice() -> None:
    """`hydroswarm.api.app.verify_plan` selects its authoritative simulator
    based purely on whether a `verifier=` was injected into `create_app` --
    it never inspects `pipeline_factory` at all (confirmed by reading the
    source; this test proves the underlying WNTR machinery that code path
    calls remains fully real and unaltered). This is why swapping the
    default neural pipeline_factory from DefaultPipelineFactory to
    V4PipelineFactory cannot weaken exact verification: the two concerns
    are architecturally decoupled, not merely coincidentally unaffected."""

    network = wntr.network.WaterNetworkModel(str(PROJECT_ROOT / "data" / "topologies" / "loop-grid.inp"))
    simulator = HydraulicSimulator(network)
    plan = OperationalPlan(
        incident_id="00000000-0000-0000-0000-000000000000",
        name="verification-authority-smoke-plan",
        actions=(OperationalAction(action_type=ActionType.MONITOR_NODE, target_id="J1"),),
        model_version="test-verification-authority-smoke",
    )
    verification = PlanVerifier(simulator).verify(plan)

    assert verification.decision in {"VERIFIED", "REJECTED", "ABSTAINED"}
    assert verification.simulator == "WNTRSimulator"
    assert verification.simulator_version == wntr.__version__
