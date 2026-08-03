from hydroswarm.domain import OODLevel
from hydroswarm.inference import OODDetector, OODReference


def test_unseen_topology_hash_enters_caution_even_with_valid_node_count() -> None:
    detector = OODDetector(
        OODReference(
            minimum_nodes=4,
            maximum_nodes=10,
            validated_network_hashes=("validated-topology",),
        )
    )

    assert detector.topology_novelty(node_count=7, network_hash="new-topology") == 1.0
    assert detector.topology_level(
        node_count=7, network_hash="new-topology"
    ) == OODLevel.CAUTION
    assert detector.topology_level(
        node_count=7, network_hash="validated-topology"
    ) == OODLevel.NORMAL
