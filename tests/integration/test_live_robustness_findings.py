"""Regression contracts for the LIVE-study authority findings."""

from __future__ import annotations

from hydroswarm.data.scenarios import network_sha256
from hydroswarm.inference import OODDetector, OODReference
from hydroswarm.simulation.wrapper import wntr


def test_ood_topology_allow_list_uses_structural_hashes() -> None:
    assert wntr is not None
    loop = wntr.network.WaterNetworkModel("data/topologies/loop-grid.inp")
    coastal = wntr.network.WaterNetworkModel("data/topologies/coastal-branch.inp")
    detector = OODDetector(OODReference(validated_network_hashes=(network_sha256(loop),)))
    assert detector.topology_novelty(node_count=len(loop.node_name_list), network_hash=network_sha256(loop)) == 0.0
    assert detector.topology_novelty(node_count=len(coastal.node_name_list), network_hash=network_sha256(coastal)) > 0.0
