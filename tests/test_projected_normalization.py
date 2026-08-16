"""Characterize projected-model normalization in exposed response coordinates."""

from math import pi

import numpy as np
import pytest

from jeam.Models.HyperSpherical import ProjectedHyperSphericalDiffusionModel
from jeam.Models.Spherical import ProjectedSphericalDiffusionModel
from tests.numerical_diagnostics import (
    first_order_quadrature_tolerance,
    fixed_zero_drift_surface_density,
    trapezoid_mass,
)

PROJECTED_MODEL_CASES = [
    pytest.param(
        3,
        ProjectedSphericalDiffusionModel,
        np.array([pi / 6, 2 * pi / 3]),
        2.0,
        id="psdm",
    ),
    pytest.param(
        4,
        ProjectedHyperSphericalDiffusionModel,
        np.array([[pi / 6, pi / 4], [2 * pi / 3, pi / 3]]),
        pi,
        id="phsdm",
    ),
]


def _projected_coordinate_jacobian(dimension: int, angles: np.ndarray) -> np.ndarray:
    """Return the response Jacobian after marginalizing the hidden azimuth."""
    if dimension == 3:
        return np.sin(angles)
    return np.sin(angles[:, 0]) ** 2 * np.sin(angles[:, 1])


@pytest.mark.parametrize(
    (
        "dimension",
        "model_type",
        "angles",
        "_response_jacobian_integral",
    ),
    PROJECTED_MODEL_CASES,
)
def test_projected_density_uses_exposed_response_coordinates(
    dimension, model_type, angles, _response_jacobian_integral
):
    decision_times = np.array([0.2, 0.8])
    observed = np.exp(
        model_type().joint_lpdf(
            rt=decision_times,
            theta=angles,
            drift_vec=np.zeros(dimension - 1),
            ndt=0.0,
            threshold=1.0,
        )
    )
    coordinate_jacobian = _projected_coordinate_jacobian(dimension, angles)
    expected = (
        2
        * pi
        * fixed_zero_drift_surface_density(dimension, decision_times)
        * coordinate_jacobian
    )

    np.testing.assert_allclose(observed, expected, rtol=2e-9, atol=1e-15)


@pytest.mark.parametrize(
    (
        "dimension",
        "model_type",
        "_angles",
        "response_jacobian_integral",
    ),
    PROJECTED_MODEL_CASES,
)
def test_projected_density_integrates_to_one(
    dimension, model_type, _angles, response_jacobian_integral
):
    decision_times = np.linspace(1e-6, 12.0, 30_000)
    if dimension == 3:
        unit_jacobian_angles = np.full(decision_times.size, pi / 2)
    else:
        unit_jacobian_angles = np.full((decision_times.size, 2), pi / 2)

    coordinate_density = np.exp(
        model_type().joint_lpdf(
            rt=decision_times,
            theta=unit_jacobian_angles,
            drift_vec=np.zeros(dimension - 1),
            ndt=0.0,
            threshold=1.0,
        )
    )
    mass = response_jacobian_integral * trapezoid_mass(
        coordinate_density, (decision_times,)
    )
    tolerance = first_order_quadrature_tolerance(
        decision_times[1] - decision_times[0], multiplier=0.25, floor=1e-5
    )

    assert mass == pytest.approx(1.0, abs=tolerance)


@pytest.mark.parametrize(
    ("model_type", "polar_angles", "drift_dimension"),
    [
        pytest.param(
            ProjectedSphericalDiffusionModel,
            np.array([0.0, pi]),
            2,
            id="psdm",
        ),
        pytest.param(
            ProjectedHyperSphericalDiffusionModel,
            np.array(
                [
                    [0.0, pi / 2],
                    [pi, pi / 2],
                    [pi / 2, 0.0],
                    [pi / 2, pi],
                ]
            ),
            3,
            id="phsdm",
        ),
    ],
)
def test_projected_coordinate_density_vanishes_at_poles(
    model_type, polar_angles, drift_dimension
):
    log_density = model_type().joint_lpdf(
        rt=np.full(polar_angles.shape[0], 0.5),
        theta=polar_angles,
        drift_vec=np.zeros(drift_dimension),
        ndt=0.0,
        threshold=1.0,
    )

    assert np.all(np.exp(log_density) < np.finfo(np.float64).eps)
