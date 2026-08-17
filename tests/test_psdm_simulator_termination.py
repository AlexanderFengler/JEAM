import numpy as np
import pytest

from jeam.Models.Spherical import ProjectedSphericalDiffusionModel
from jeam.utility.simulators import (
    simulate_custom_threshold_PSDM_trial,
    simulate_PSDM_trial,
)


@pytest.mark.parametrize(
    "threshold_dynamic", ["fixed", "linear", "exponential", "hyperbolic"]
)
def test_builtin_helper_returns_nan_pair_when_the_boundary_is_not_reached(
    threshold_dynamic,
):
    rt, response = simulate_PSDM_trial(
        threshold=1.0,
        drift_vec=np.array([0.0, 0.0]),
        ndt=0.2,
        threshold_dynamic=threshold_dynamic,
        decay=0.2,
        sigma=0.0,
        dt=0.01,
        max_time=0.03,
        random_state=1947,
    )

    assert np.isnan(rt)
    assert np.isnan(response)


def test_batch_preserves_omitted_row_and_completed_row():
    simulated = ProjectedSphericalDiffusionModel().simulate(
        drift_vec=np.array([[0.0, 0.0], [10.0, 0.0]]),
        ndt=0.2,
        threshold=0.1,
        sigma=0.0,
        dt=0.01,
        max_time=0.03,
        n_sample=2,
        random_state=1947,
    )

    assert simulated.iloc[0].isna().all()
    assert simulated.loc[1, "rt"] == pytest.approx(0.21)
    assert simulated.loc[1, "response"] == 0.0


def test_custom_threshold_returns_nan_pair_at_the_same_horizon():
    rt, response = simulate_custom_threshold_PSDM_trial(
        threshold_function=lambda time: 1.0,
        drift_vec=np.array([0.0, 0.0]),
        ndt=0.2,
        sigma=0.0,
        dt=0.01,
        max_time=0.03,
        random_state=1947,
    )

    assert np.isnan(rt)
    assert np.isnan(response)


def test_horizon_does_not_change_a_completed_seeded_trial():
    arguments = {
        "threshold": 0.45,
        "drift_vec": np.array([0.7, 0.35]),
        "ndt": 0.2,
        "sigma": 0.8,
        "dt": 0.005,
        "random_state": 1947,
    }

    default_horizon = simulate_PSDM_trial(**arguments)
    explicit_horizon = simulate_PSDM_trial(**arguments, max_time=20.0)

    np.testing.assert_allclose(
        default_horizon,
        (0.41500000000000015, 1.3586944568046324),
        rtol=1e-13,
        atol=1e-15,
    )
    assert default_horizon == explicit_horizon


def test_last_step_is_shortened_to_the_physical_horizon():
    rt, response = simulate_PSDM_trial(
        threshold=0.04,
        drift_vec=np.array([10.0, 0.0]),
        ndt=0.2,
        sigma=0.0,
        dt=0.01,
        max_time=0.005,
        random_state=1947,
    )

    assert rt == pytest.approx(0.205)
    assert response == 0.0


@pytest.mark.parametrize("max_time", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_max_time_is_rejected(max_time):
    with pytest.raises(ValueError, match="max_time"):
        simulate_PSDM_trial(
            threshold=1.0,
            drift_vec=np.array([0.0, 0.0]),
            ndt=0.2,
            max_time=max_time,
            random_state=1947,
        )


@pytest.mark.parametrize("dt", [0.0, -0.01, np.inf, -np.inf, np.nan])
def test_invalid_dt_is_rejected(dt):
    with pytest.raises(ValueError, match="dt"):
        simulate_PSDM_trial(
            threshold=1.0,
            drift_vec=np.array([0.0, 0.0]),
            ndt=0.2,
            dt=dt,
            max_time=0.03,
            random_state=1947,
        )
