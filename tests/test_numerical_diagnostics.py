"""Tests for the independent numerical diagnostic layer."""

from math import pi

import mpmath as mp
import numpy as np
import pytest
from scipy.special import ive

from tests.numerical_diagnostics import (
    empirical_cdf_max_error,
    first_order_quadrature_tolerance,
    fixed_zero_drift_surface_density,
    scaled_bessel_i_reference,
    trapezoid_mass,
    unit_sphere_surface_jacobian,
)


def test_trapezoid_mass_integrates_rectangular_linear_density_exactly():
    x = np.linspace(0.0, 1.0, 17)
    y = np.linspace(0.0, 2.0, 19)
    values = x[:, np.newaxis] + y[np.newaxis, :]

    assert trapezoid_mass(values, (x, y)) == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("axes", "expected_area"),
    [
        ((np.linspace(-pi, pi, 1_001),), 2 * pi),
        (
            (
                np.linspace(0.0, pi, 301),
                np.linspace(-pi, pi, 401),
            ),
            4 * pi,
        ),
        (
            (
                np.linspace(0.0, pi, 101),
                np.linspace(0.0, pi, 101),
                np.linspace(-pi, pi, 151),
            ),
            2 * pi**2,
        ),
    ],
)
def test_unit_sphere_jacobian_integrates_to_known_surface_area(axes, expected_area):
    angular_grid = np.meshgrid(*axes, indexing="ij")
    jacobian = unit_sphere_surface_jacobian(*angular_grid)

    assert trapezoid_mass(jacobian, axes) == pytest.approx(expected_area, rel=2e-4)


@pytest.mark.parametrize("dimension", [2, 3, 4])
def test_fixed_zero_drift_reference_is_positive_and_vectorized(dimension):
    times = np.array([[0.1, 0.3], [0.7, 1.5]])

    density = fixed_zero_drift_surface_density(dimension, times)

    assert density.shape == times.shape
    assert density.dtype == np.float64
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)


@pytest.mark.parametrize(
    ("order", "argument"),
    [(0.0, 1.25), (1.5, 25.0), (0.0, 1_000.0)],
)
def test_scaled_bessel_reference_matches_scipy(order, argument):
    expected = scaled_bessel_i_reference(order, argument)

    assert float(expected) == pytest.approx(ive(order, argument), rel=2e-14)
    assert isinstance(expected, mp.mpf)


def test_empirical_cdf_error_handles_deterministic_uniform_quantiles():
    n_samples = 100
    samples = (np.arange(n_samples) + 0.5) / n_samples

    error = empirical_cdf_max_error(samples, lambda values: values)

    assert error == pytest.approx(0.5 / n_samples)


def test_first_order_quadrature_tolerance_tracks_approximation_step():
    assert first_order_quadrature_tolerance(0.01) == pytest.approx(0.05)
    assert first_order_quadrature_tolerance(1e-9) == pytest.approx(1e-6)


@pytest.mark.parametrize(
    ("diagnostic_call", "message"),
    [
        (
            lambda: trapezoid_mass(np.ones((2, 2)), (np.arange(2),)),
            "does not match integration grid",
        ),
        (
            lambda: trapezoid_mass(np.ones(2), (np.ones((2, 1)),)),
            "integration axis",
        ),
        (lambda: unit_sphere_surface_jacobian(), "angular coordinate"),
        (
            lambda: fixed_zero_drift_surface_density(5, [0.5]),
            "supports dimensions",
        ),
        (
            lambda: fixed_zero_drift_surface_density(2, [0.5], threshold=0.0),
            "must be positive",
        ),
        (
            lambda: fixed_zero_drift_surface_density(2, [0.0]),
            "times must be positive",
        ),
        (
            lambda: scaled_bessel_i_reference(0.0, 1.0, decimal_places=15),
            "at least 16",
        ),
        (
            lambda: empirical_cdf_max_error([], lambda values: values),
            "finite values",
        ),
        (
            lambda: empirical_cdf_max_error([0.5], lambda values: np.array([1.5])),
            "one probability",
        ),
        (
            lambda: first_order_quadrature_tolerance(0.0),
            "step and multiplier",
        ),
    ],
)
def test_numerical_diagnostics_reject_invalid_inputs(diagnostic_call, message):
    with pytest.raises(ValueError, match=message):
        diagnostic_call()
