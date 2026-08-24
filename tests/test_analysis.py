from pathlib import Path

import pytest

from ai4us.analysis.perception_state_replacement import recompute


ROOT = Path(__file__).resolve().parents[1]


def test_released_state_replacement_recomputation() -> None:
    result = recompute(ROOT / "data/processed/model_outputs")
    assert result["registered_and_usable_pair_units"] == {
        "behavior": [216, 216],
        "state_replacement": [216, 216],
    }
    assert result["equal_norm_checks_passed"] == 216
    assert result["clean_profile_reference_correlation"]["n_scenes"] == 12
    assert result["clean_profile_reference_correlation"]["estimate"] == pytest.approx(
        0.8474393841800834, abs=1e-12
    )
