"""Known-good numerical baselines from the JEAM paper audit."""

from math import pi

import numpy as np
import pytest

from jeam.Models.Circular import CircularDiffusionModel
from jeam.Models.HyperSpherical import HyperSphericalDiffusionModel
from jeam.Models.Spherical import SphericalDiffusionModel
from jeam.utility.fpts import ie_fpt_linear
from tests.numerical_diagnostics import (
    first_order_quadrature_tolerance,
    fixed_zero_drift_surface_density,
    trapezoid_mass,
)

FIXED_MODEL_CASES = [
    pytest.param(2, CircularDiffusionModel, 2 * pi, id="cdm"),
    pytest.param(3, SphericalDiffusionModel, 4 * pi, id="sdm"),
    pytest.param(4, HyperSphericalDiffusionModel, 2 * pi**2, id="hsdm"),
]


def _unit_jacobian_angles(n_observations: int, dimension: int) -> np.ndarray:
    """Choose angles where natural-surface and coordinate Jacobians both equal one."""
    angles = np.full((n_observations, dimension - 1), pi / 2, dtype=np.float64)
    return angles[:, 0] if dimension == 2 else angles


@pytest.mark.parametrize(
    ("dimension", "model_type", "_surface_area"), FIXED_MODEL_CASES
)
def test_fixed_zero_drift_density_matches_independent_bessel_series(
    dimension, model_type, _surface_area
):
    decision_times = np.array([0.05, 0.2, 0.8, 2.0])
    threshold = 1.3
    sigma = 0.8
    angles = _unit_jacobian_angles(decision_times.size, dimension)

    observed = np.exp(
        model_type().joint_lpdf(
            rt=decision_times,
            theta=angles,
            drift_vec=np.zeros(dimension),
            ndt=0.0,
            threshold=threshold,
            sigma=sigma,
        )
    )
    expected = fixed_zero_drift_surface_density(
        dimension,
        decision_times,
        threshold=threshold,
        sigma=sigma,
    )

    np.testing.assert_allclose(observed, expected, rtol=2e-9, atol=1e-15)


@pytest.mark.parametrize(("dimension", "model_type", "surface_area"), FIXED_MODEL_CASES)
def test_fixed_zero_drift_radial_mass_is_one(dimension, model_type, surface_area):
    decision_times = np.linspace(1e-6, 12.0, 30_000)
    angles = _unit_jacobian_angles(decision_times.size, dimension)
    surface_density = np.exp(
        model_type().joint_lpdf(
            rt=decision_times,
            theta=angles,
            drift_vec=np.zeros(dimension),
            ndt=0.0,
            threshold=1.0,
        )
    )
    mass = surface_area * trapezoid_mass(surface_density, (decision_times,))
    tolerance = first_order_quadrature_tolerance(
        decision_times[1] - decision_times[0], multiplier=0.25, floor=1e-5
    )

    assert mass == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize(
    ("dimension", "audited_mass"),
    [
        (2, 1.000000913621579),
        (3, 1.000000007361337),
        (4, 0.9999989759171664),
    ],
)
def test_linear_integral_equation_preserves_audited_mass(dimension, audited_mass):
    approximation_step = 0.005
    density, times = ie_fpt_linear(
        threshold=3.0,
        decay=0.5,
        q=dimension,
        z=1e-6,
        sigma=2.0,
        dt=approximation_step,
        T_max=6.0,
    )
    mass = trapezoid_mass(density, (times,))
    invariant_tolerance = first_order_quadrature_tolerance(
        approximation_step, multiplier=1e-3, floor=2e-6
    )

    assert density.shape == times.shape == (1_202,)
    assert times[1] - times[0] == pytest.approx(approximation_step)
    assert density.min() >= -invariant_tolerance
    assert mass == pytest.approx(audited_mass, abs=5e-7)
    assert mass == pytest.approx(1.0, abs=invariant_tolerance)
