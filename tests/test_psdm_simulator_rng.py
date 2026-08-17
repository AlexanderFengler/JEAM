import numpy as np
import pandas as pd
import pytest

from jeam.Models.Spherical import ProjectedSphericalDiffusionModel
from jeam.utility.simulators import simulate_PSDM_trial


def _simulate_fixed(random_state, *, n_sample=8):
    return ProjectedSphericalDiffusionModel().simulate(
        drift_vec=np.array([0.7, 0.35]),
        ndt=0.2,
        threshold=0.45,
        s_v=0.1,
        s_t=0.05,
        sigma=0.8,
        dt=0.005,
        n_sample=n_sample,
        random_state=random_state,
    )


def _assert_legacy_state_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_fixed_psdm_same_integer_seed_reproduces_exact_batch():
    first = _simulate_fixed(1947)
    second = _simulate_fixed(1947)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_single_trial_helper_accepts_the_same_seed_contract():
    arguments = (0.45, np.array([0.7, 0.35]), 0.2)

    first = simulate_PSDM_trial(*arguments, dt=0.005, random_state=1947)
    second = simulate_PSDM_trial(*arguments, dt=0.005, random_state=1947)

    assert first == second


def test_fixed_psdm_different_seeds_change_batch():
    first = _simulate_fixed(1947)
    second = _simulate_fixed(1948)

    assert not first.equals(second)


def test_fresh_generators_reproduce_and_reused_generator_advances():
    first_rng = np.random.default_rng(1947)
    second_rng = np.random.default_rng(1947)

    first = _simulate_fixed(first_rng)
    matching = _simulate_fixed(second_rng)
    advanced = _simulate_fixed(first_rng)

    pd.testing.assert_frame_equal(first, matching, check_exact=True)
    assert not first.equals(advanced)


def test_one_rng_stream_is_shared_across_trials():
    simulated = _simulate_fixed(1947, n_sample=6)

    assert len(simulated.drop_duplicates()) > 1


@pytest.mark.parametrize("random_state", [1947, None])
def test_psdm_simulation_does_not_mutate_legacy_numpy_state(random_state):
    original_state = np.random.get_state()
    try:
        np.random.seed(831)
        before = np.random.get_state()

        _simulate_fixed(random_state)

        after = np.random.get_state()
        _assert_legacy_state_equal(before, after)
    finally:
        np.random.set_state(original_state)


def test_fixed_psdm_accepts_trialwise_parameters_and_preserves_output_support():
    n_sample = 4
    simulated = ProjectedSphericalDiffusionModel().simulate(
        drift_vec=np.array([[0.7, 0.1], [0.4, 0.2], [-0.2, 0.5], [0.1, 0.6]]),
        ndt=np.array([0.1, 0.12, 0.14, 0.16]),
        threshold=np.array([0.35, 0.4, 0.45, 0.5]),
        n_sample=n_sample,
        dt=0.005,
        random_state=519,
    )

    assert simulated.shape == (n_sample, 2)
    assert list(simulated.columns) == ["rt", "response"]
    assert np.all(simulated["rt"].to_numpy() > 0)
    assert np.all(simulated["response"].to_numpy() >= 0)
    assert np.all(simulated["response"].to_numpy() <= np.pi)


def test_psdm_includes_the_positive_pi_boundary():
    simulated = ProjectedSphericalDiffusionModel().simulate(
        drift_vec=np.array([-1.0, 0.0]),
        ndt=0.1,
        threshold=0.1,
        sigma=0.0,
        dt=0.01,
        random_state=519,
    )

    assert simulated.loc[0, "response"] == np.pi


@pytest.mark.parametrize("threshold_dynamic", ["linear", "exponential", "hyperbolic"])
def test_collapsing_psdm_thresholds_use_the_seeded_stream(threshold_dynamic):
    model = ProjectedSphericalDiffusionModel(threshold_dynamic=threshold_dynamic)
    arguments = {
        "drift_vec": np.array([0.5, 0.25]),
        "ndt": 0.15,
        "threshold": 0.5,
        "decay": 0.2,
        "n_sample": 4,
        "dt": 0.005,
        "random_state": 997,
    }

    first = model.simulate(**arguments)
    second = model.simulate(**arguments)

    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_custom_threshold_psdm_same_seed_reproduces_exact_batch():
    model = ProjectedSphericalDiffusionModel(threshold_dynamic="custom")

    def threshold_function(time):
        return 0.5 / (1 + 0.2 * time)

    first = model.simulate(
        drift_vec=np.array([0.5, 0.25]),
        ndt=0.15,
        threshold_function=threshold_function,
        n_sample=5,
        dt=0.005,
        random_state=997,
    )
    second = model.simulate(
        drift_vec=np.array([0.5, 0.25]),
        ndt=0.15,
        threshold_function=threshold_function,
        n_sample=5,
        dt=0.005,
        random_state=997,
    )

    pd.testing.assert_frame_equal(first, second, check_exact=True)
