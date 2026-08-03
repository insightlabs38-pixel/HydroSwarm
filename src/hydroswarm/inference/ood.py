"""Deterministic multi-signal out-of-distribution assessment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from hydroswarm.classical import EstimatedHydraulicState
from hydroswarm.domain import OODLevel
from hydroswarm.preprocessing import SensorSeries


@dataclass(frozen=True, slots=True)
class OODComponents:
    latent_distance: float
    energy: float
    network_novelty: float
    demand_shift: float
    sensor_consistency: float
    combined: float


@dataclass(frozen=True, slots=True)
class OODReference:
    latent_center: tuple[float, ...] = ()
    latent_scale: float = 1.0
    minimum_nodes: int = 1
    maximum_nodes: int = 10_000
    caution_threshold: float = 0.45
    outside_threshold: float = 0.75


class OODDetector:
    def __init__(self, reference: OODReference | None = None) -> None:
        self.reference = reference or OODReference()

    @staticmethod
    def _clip(value: float) -> float:
        return float(np.clip(value, 0.0, 1.0))

    def evaluate(
        self,
        *,
        node_count: int,
        state: EstimatedHydraulicState,
        sensor_series: Sequence[SensorSeries],
        latent: np.ndarray | None = None,
        neural_probabilities: np.ndarray | None = None,
        ood_probabilities: np.ndarray | None = None,
    ) -> tuple[OODComponents, OODLevel]:
        if latent is not None and self.reference.latent_center:
            center = np.asarray(self.reference.latent_center, dtype=float)
            flattened = np.asarray(latent, dtype=float).reshape(-1)
            if flattened.size == center.size:
                latent_distance = self._clip(
                    np.linalg.norm(flattened - center)
                    / max(self.reference.latent_scale * np.sqrt(center.size), 1e-9)
                )
            else:
                latent_distance = 1.0
        else:
            latent_distance = 0.0
        if ood_probabilities is not None and len(ood_probabilities) >= 3:
            energy = self._clip(float(ood_probabilities[-1]))
        elif neural_probabilities is not None:
            energy = self._clip(1.0 - float(np.max(neural_probabilities)))
        else:
            energy = 0.25
        if node_count < self.reference.minimum_nodes:
            network_novelty = self._clip(
                (self.reference.minimum_nodes - node_count) / max(1, self.reference.minimum_nodes)
            )
        elif node_count > self.reference.maximum_nodes:
            network_novelty = self._clip(
                (node_count - self.reference.maximum_nodes) / max(1, self.reference.maximum_nodes)
            )
        else:
            network_novelty = 0.0
        demand_shift = self._clip(float(state.residuals.mismatch_scores.get("demand", 0.0)))
        qualities = [item.health[-1] for item in sensor_series]
        missing = [float(item.missing[-1]) for item in sensor_series]
        sensor_consistency = self._clip(
            1.0 - (sum(qualities) / len(qualities)) + (sum(missing) / len(missing)) * 0.5
        ) if sensor_series else 1.0
        combined = self._clip(
            0.25 * latent_distance
            + 0.20 * energy
            + 0.15 * network_novelty
            + 0.20 * demand_shift
            + 0.20 * sensor_consistency
        )
        components = OODComponents(
            latent_distance=latent_distance,
            energy=energy,
            network_novelty=network_novelty,
            demand_shift=demand_shift,
            sensor_consistency=sensor_consistency,
            combined=combined,
        )
        if combined >= self.reference.outside_threshold:
            level = OODLevel.OUTSIDE_VALIDATED_RANGE
        elif combined >= self.reference.caution_threshold:
            level = OODLevel.CAUTION
        else:
            level = OODLevel.NORMAL
        return components, level
