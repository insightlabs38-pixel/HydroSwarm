"""DefaultPipelineFactory composes the actual promoted checkpoint,
calibration artifact, and (now) normalization identity used in production.
No prior test exercised it directly -- these are the tests that would have
caught core-issues.txt repair item 2's real regression (the promoted
checkpoint's source_region_head became shape-incompatible) before it ever
reached a real deployment."""

from __future__ import annotations

from pathlib import Path

from hydroswarm.runtime.defaults import DefaultPipelineFactory
from hydroswarm.simulation.wrapper import wntr


def test_promoted_checkpoint_loads_successfully() -> None:
    factory = DefaultPipelineFactory()
    assert factory.trained_assets_ready is True
    assert factory.fallback_reason is None


def test_promoted_checkpoint_required_the_known_v2_source_region_migration() -> None:
    """Documents, rather than hides, that the currently promoted checkpoint
    predates repair item 2's architecture fix -- exactly the two
    source_region_head parameters, nothing else, were re-initialized
    rather than loaded from the checkpoint's own (shape-incompatible,
    never-meaningfully-trained) weights."""

    factory = DefaultPipelineFactory()
    assert factory.trained_assets_ready is True
    assert set(factory.migrated_parameters) == {
        "source_region_head.network.1.weight",
        "source_region_head.network.1.bias",
    }


def test_promoted_checkpoint_calibration_still_validates_with_no_normalization() -> None:
    """core-issues.txt repair item 6: the promoted calibration artifact was
    fit with no governed normalization layer (the only state that has ever
    existed), and the runtime feature builder also defaults to none -- they
    must still agree, not silently diverge because one side started
    recording an explicit identity and the other did not."""

    factory = DefaultPipelineFactory()
    factory._load_assets()
    assert factory._calibrator is not None
    assert factory._feature_builder is not None
    from hydroswarm.preprocessing.builder import NO_NORMALIZATION_SENTINEL

    assert factory._calibrator.artifact.normalization_hash == NO_NORMALIZATION_SENTINEL
    assert factory._feature_builder.normalization_fingerprint == NO_NORMALIZATION_SENTINEL


def test_promoted_checkpoint_declares_only_sentinel_as_a_trained_task() -> None:
    """core-issues.txt repair item 8: the real promoted checkpoint's own
    metadata.json now declares trained_tasks, and the real hybrid pipeline
    built from it must actually gate Scout/Strategist/OOD accordingly --
    not just the constant in isolation."""

    factory = DefaultPipelineFactory()
    assert factory.trained_assets_ready is True
    assert factory.trained_tasks == frozenset({"sentinel"})

    if wntr is None:
        return
    network_path = Path(__file__).resolve().parents[2] / "data" / "frozen" / "golden_network.inp"
    pipeline = factory(None, network_path)
    assert pipeline.trained_tasks == frozenset({"sentinel"})
