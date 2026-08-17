"""Scientific consistency gates for the fixed projected spherical model."""

from math import pi

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import differential_evolution
from scipy.special import iv

from jeam.Models.Spherical import ProjectedSphericalDiffusionModel
from tests.numerical_diagnostics import fixed_psdm_coordinate_density


def _fixed_likelihood_reference_summary(
    model,
    *,
    drift,
    threshold,
    sigma,
    ndt,
    quantile_probabilities,
):
    """Obtain RT quantiles and polar moments from normalized likelihood slices."""
    decision_times = np.linspace(1e-6, 12.0, 60_000)
    reference_angle = pi / 2
    joint_time_slice = np.exp(
        model.joint_lpdf(
            rt=decision_times + ndt,
            theta=np.full(decision_times.size, reference_angle),
            drift_vec=drift,
            ndt=ndt,
            threshold=threshold,
            sigma=sigma,
        )
    )
    rt_density = joint_time_slice / trapezoid(joint_time_slice, x=decision_times)
    rt_cdf = cumulative_trapezoid(
        rt_density,
        x=decision_times,
        initial=0.0,
    )
    rt_cdf /= rt_cdf[-1]
    rt_quantiles = np.interp(quantile_probabilities, rt_cdf, decision_times) + ndt
    density_at_quantiles = np.interp(
        rt_quantiles - ndt,
        decision_times,
        rt_density,
    )

    angles = np.linspace(0.0, pi, 4_001)
    joint_angle_slice = np.exp(
        model.joint_lpdf(
            rt=np.full(angles.size, ndt + 0.5),
            theta=angles,
            drift_vec=drift,
            ndt=ndt,
            threshold=threshold,
            sigma=sigma,
        )
    )
    angle_density = joint_angle_slice / trapezoid(joint_angle_slice, x=angles)
    polar_moments = np.array(
        [
            trapezoid(np.cos(angles) ** power * angle_density, x=angles)
            for power in (1, 2)
        ]
    )
    return rt_quantiles, density_at_quantiles, polar_moments


def test_fixed_psdm_matches_independent_asymmetric_drift_oracle():
    decision_times = np.array([0.08, 0.25, 0.9, 2.0])
    angles = np.array([0.2, 0.8, 1.6, 2.7])
    drift = np.array([0.65, 1.1])
    threshold = 1.35
    sigma = 0.8
    ndt = 0.22

    observed = np.exp(
        ProjectedSphericalDiffusionModel().joint_lpdf(
            rt=decision_times + ndt,
            theta=angles,
            drift_vec=drift,
            ndt=ndt,
            threshold=threshold,
            sigma=sigma,
        )
    )
    expected = fixed_psdm_coordinate_density(
        decision_times,
        angles,
        drift,
        threshold=threshold,
        sigma=sigma,
    )

    np.testing.assert_allclose(observed, expected, rtol=2e-9, atol=1e-15)


def test_fixed_psdm_asymmetric_density_integrates_to_one():
    decision_times = np.linspace(1e-6, 12.0, 30_000)
    angles = np.linspace(0.0, pi, 4_001)
    drift = np.array([0.6, 1.0])
    threshold = 1.1
    sigma = 0.9
    reference_angle = pi / 2

    density_at_reference = np.exp(
        ProjectedSphericalDiffusionModel().joint_lpdf(
            rt=decision_times,
            theta=np.full(decision_times.size, reference_angle),
            drift_vec=drift,
            ndt=0.0,
            threshold=threshold,
            sigma=sigma,
        )
    )
    axial_drift, radial_drift = drift
    angular_ratio = (
        np.sin(angles)
        * np.exp(threshold * axial_drift * np.cos(angles) / sigma**2)
        * iv(0, threshold * radial_drift * np.sin(angles) / sigma**2)
        / iv(0, threshold * radial_drift / sigma**2)
    )
    mass = trapezoid(density_at_reference, x=decision_times) * trapezoid(
        angular_ratio, x=angles
    )

    assert mass == pytest.approx(1.0, abs=1e-5)


def test_fixed_psdm_simulator_matches_likelihood_quantiles_and_moments():
    model = ProjectedSphericalDiffusionModel()
    drift = np.array([0.6, 1.0])
    threshold = 1.1
    sigma = 0.9
    ndt = 0.2
    quantile_probabilities = np.array([0.1, 0.5, 0.9])
    n_sample = 30_000
    simulation_step = 0.0005

    expected_quantiles, density_at_quantiles, expected_polar_moments = (
        _fixed_likelihood_reference_summary(
            model,
            drift=drift,
            threshold=threshold,
            sigma=sigma,
            ndt=ndt,
            quantile_probabilities=quantile_probabilities,
        )
    )
    simulated = model.simulate(
        drift_vec=drift,
        ndt=ndt,
        threshold=threshold,
        sigma=sigma,
        dt=simulation_step,
        n_sample=n_sample,
        random_state=1947,
    )

    assert not simulated.isna().any().any()
    observed_quantiles = np.quantile(simulated["rt"], quantile_probabilities)
    quantile_standard_errors = (
        np.sqrt(quantile_probabilities * (1 - quantile_probabilities) / n_sample)
        / density_at_quantiles
    )
    # Four Monte Carlo standard errors plus one reporting time step keeps the gate
    # sensitive to scientific drift while accounting for the Euler time grid.
    quantile_error_budget = 4 * quantile_standard_errors + simulation_step
    np.testing.assert_array_less(
        np.abs(observed_quantiles - expected_quantiles),
        quantile_error_budget,
    )

    cosine = np.cos(simulated["response"].to_numpy())
    observed_polar_statistics = np.column_stack((cosine, cosine**2))
    observed_polar_moments = observed_polar_statistics.mean(axis=0)
    moment_standard_errors = observed_polar_statistics.std(axis=0, ddof=1) / np.sqrt(
        n_sample
    )
    np.testing.assert_array_less(
        np.abs(observed_polar_moments - expected_polar_moments),
        4 * moment_standard_errors,
    )


@pytest.mark.slow
def test_fixed_psdm_recovers_asymmetric_drift_threshold_and_ndt():
    model = ProjectedSphericalDiffusionModel()
    # Parameter order: axial drift, projected radial drift, threshold, NDT.
    true_parameters = np.array([0.6, 1.0, 1.1, 0.2])
    simulated = model.simulate(
        drift_vec=true_parameters[:2],
        ndt=true_parameters[3],
        threshold=true_parameters[2],
        sigma=1.0,
        s_v=0.0,
        s_t=0.0,
        dt=0.0005,
        n_sample=4_000,
        random_state=4801,
    )
    rt = simulated["rt"].to_numpy()
    response = simulated["response"].to_numpy()

    def negative_log_likelihood(parameters):
        log_density = model.joint_lpdf(
            rt=rt,
            theta=response,
            drift_vec=parameters[:2],
            ndt=parameters[3],
            threshold=parameters[2],
            sigma=1.0,
            s_v=0.0,
            s_t=0.0,
        )
        if not np.all(np.isfinite(log_density)):
            return np.inf
        return -float(np.sum(log_density))

    bounds = [
        (-0.5, 1.5),
        (0.1, 2.0),
        (0.6, 1.6),
        (0.05, float(rt.min() - 1e-6)),
    ]
    recovery = differential_evolution(
        negative_log_likelihood,
        bounds=bounds,
        seed=15,
        maxiter=60,
        popsize=8,
        tol=1e-4,
        polish=True,
        workers=1,
        updating="immediate",
    )

    assert recovery.success, recovery.message
    assert recovery.fun <= negative_log_likelihood(true_parameters)
    absolute_errors = np.abs(recovery.x - true_parameters)
    np.testing.assert_array_less(
        absolute_errors,
        np.array([0.12, 0.15, 0.05, 0.015]),
    )
