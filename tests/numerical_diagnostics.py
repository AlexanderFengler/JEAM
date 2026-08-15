"""Independent numerical oracles for JEAM characterization tests.

Nothing in this module imports JEAM. Keeping the diagnostics independent prevents a
production implementation from serving as its own numerical reference.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import pi

import mpmath as mp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid
from scipy.special import jn_zeros, jv


def trapezoid_mass(values: ArrayLike, axes: Sequence[ArrayLike]) -> float:
    """Integrate values over a rectangular grid using one axis per dimension."""
    result = np.asarray(values, dtype=np.float64)
    coordinates = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    expected_shape = tuple(axis.size for axis in coordinates)

    if result.shape != expected_shape:
        raise ValueError(
            f"values shape {result.shape} does not match integration grid "
            f"{expected_shape}"
        )
    if any(axis.ndim != 1 or axis.size < 2 for axis in coordinates):
        raise ValueError("each integration axis must be one-dimensional with 2+ points")

    for axis in reversed(coordinates):
        result = trapezoid(result, x=axis, axis=-1)
    return float(result)


def unit_sphere_surface_jacobian(*angles: ArrayLike) -> NDArray[np.float64]:
    """Return the standard unit-sphere Jacobian for broadcast angular arrays."""
    if not angles:
        raise ValueError("at least one angular coordinate is required")

    broadcast = np.broadcast_arrays(
        *(np.asarray(angle, dtype=np.float64) for angle in angles)
    )
    jacobian = np.ones(broadcast[0].shape, dtype=np.float64)
    n_angles = len(broadcast)
    for index, angle in enumerate(broadcast[:-1]):
        jacobian *= np.sin(angle) ** (n_angles - index - 1)
    return jacobian


def fixed_zero_drift_surface_density(
    dimension: int,
    times: ArrayLike,
    *,
    threshold: float = 1.0,
    sigma: float = 1.0,
    n_terms: int = 100,
) -> NDArray[np.float64]:
    """Evaluate the paper's fixed-boundary joint density per unit surface area.

    The returned value is the zero-drift density over decision time and the natural
    surface measure on the boundary of a ``dimension``-dimensional ball.
    """
    if dimension not in (2, 3, 4):
        raise ValueError("the JEAM fixed-boundary oracle supports dimensions 2, 3, 4")
    if threshold <= 0 or sigma <= 0 or n_terms < 1:
        raise ValueError("threshold, sigma, and n_terms must be positive")

    time_array = np.asarray(times, dtype=np.float64)
    if np.any(time_array <= 0):
        raise ValueError("decision times must be positive")

    bessel_order = dimension / 2 - 1
    if dimension == 3:
        roots = np.arange(1, n_terms + 1, dtype=np.float64) * pi
    else:
        roots = jn_zeros(int(bessel_order), n_terms)

    coefficients = roots ** (dimension / 2) / jv(dimension / 2, roots)
    rate = roots**2 * sigma**2 / (2 * threshold**2)
    radial_series = (
        sigma**2
        / threshold**2
        * np.sum(
            coefficients[:, np.newaxis] * np.exp(-np.outer(rate, time_array.ravel())),
            axis=0,
        )
    )
    surface_density = radial_series / (2 * pi) ** (dimension / 2)
    return surface_density.reshape(time_array.shape)


def scaled_bessel_i_reference(
    order: float, argument: float, *, decimal_places: int = 80
) -> mp.mpf:
    """Evaluate ``exp(-abs(x)) * I_order(x)`` at arbitrary precision."""
    if decimal_places < 16:
        raise ValueError("decimal_places must be at least 16")
    with mp.workdps(decimal_places):
        value = mp.exp(-abs(argument)) * mp.besseli(order, argument)
    return value


def empirical_cdf_max_error(
    samples: ArrayLike,
    reference_cdf: Callable[[NDArray[np.float64]], ArrayLike],
) -> float:
    """Return the two-sided maximum empirical-CDF error."""
    ordered = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    if ordered.size == 0 or not np.all(np.isfinite(ordered)):
        raise ValueError("samples must contain finite values")

    expected = np.asarray(reference_cdf(ordered), dtype=np.float64)
    if expected.shape != ordered.shape or np.any((expected < 0) | (expected > 1)):
        raise ValueError("reference_cdf must return one probability per sample")

    lower_empirical = np.arange(ordered.size, dtype=np.float64) / ordered.size
    upper_empirical = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
    return float(
        max(
            np.max(np.abs(expected - lower_empirical)),
            np.max(np.abs(upper_empirical - expected)),
        )
    )


def first_order_quadrature_tolerance(
    approximation_step: float, *, multiplier: float = 5.0, floor: float = 1e-6
) -> float:
    """Return an explicit tolerance for a first-order time discretization."""
    if approximation_step <= 0 or multiplier <= 0 or floor < 0:
        raise ValueError(
            "step and multiplier must be positive; floor must be nonnegative"
        )
    return max(floor, multiplier * approximation_step)
