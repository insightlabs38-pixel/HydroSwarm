from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/hydrocore_v5"))
import run_m10_5a_selection as m  # noqa: E402

def test_historical_seed_order_is_identical_through_m10() -> None:
    a = m.audit()
    assert a["orders_identical"] and a["historical_precedes_m10_4"]


def test_selection_is_unique_and_independent_of_m10_4_performance() -> None:
    a = m.audit()
    assert a["selected_seed"] == 20260814 and not a["m10_4_performance_inspected"]


def test_selected_final_step_checkpoint_hash() -> None:
    a = m.audit()
    assert a["checkpoint_sha_matches"] and a["all_peers_use_final_step"]
