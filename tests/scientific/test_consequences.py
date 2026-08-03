import pandas as pd
import pytest

from hydroswarm.domain import ConsequenceMetrics
from hydroswarm.simulation.consequences import (
    PlanOutcome,
    calculate_exposure_consequences,
    rank_plan_outcomes,
)


def test_exposure_service_and_pressure_are_integrated() -> None:
    times = [0, 60, 120]
    concentration = pd.DataFrame({"J1": [0.0, 2.0, 1.0], "J2": [0.0, 0.0, 0.0]}, index=times)
    delivered = pd.DataFrame({"J1": [0.001] * 3, "J2": [0.001] * 3}, index=times)
    requested = pd.DataFrame({"J1": [0.002] * 3, "J2": [0.001] * 3}, index=times)
    pressure = pd.DataFrame({"J1": [20.0, 8.0, 12.0], "J2": [20.0] * 3}, index=times)
    metrics = calculate_exposure_consequences(
        concentration, delivered, threshold_mg_l=0.5, population_by_node={"J1": 100, "J2": 50},
        pipe_endpoints={"P1": ("J1", "J2", 250.0)}, pressure_m=pressure,
        requested_demand_m3s=requested, operation_count=2,
    )
    assert metrics.contaminant_mass_consumed_mg == pytest.approx(180.0)
    assert metrics.volume_above_threshold_l == pytest.approx(120.0)
    assert metrics.population_impacted == 100
    assert metrics.contaminated_pipe_extent_m == 250.0
    assert metrics.pressure_violation_minutes == 1.0
    assert metrics.unserved_demand_l == pytest.approx(180.0)
    assert metrics.service_availability == pytest.approx(2 / 3)


def metric(mass, pressure=0.0, service=1.0, operations=1):
    return ConsequenceMetrics(
        contaminant_mass_consumed_mg=mass, minimum_pressure_m=15.0,
        pressure_violation_minutes=pressure, service_availability=service,
        operation_count=operations,
    )


def test_pareto_ranking_keeps_tradeoffs_and_recommends_best() -> None:
    ranked = rank_plan_outcomes(
        metric(100, service=1.0, operations=0),
        [PlanOutcome("safe", metric(20)), PlanOutcome("unsafe", metric(5, pressure=60)),
         PlanOutcome("dominated", metric(40, operations=3))],
    )
    assert ranked[0].plan_id == "safe"
    assert ranked[0].recommended
    assert next(item for item in ranked if item.plan_id == "dominated").dominated
    assert ranked[0].exposure_reduction == 80
