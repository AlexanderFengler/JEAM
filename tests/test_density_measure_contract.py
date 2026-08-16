"""Characterize JEAM densities in the angular coordinates exposed to users."""

from math import pi

import numpy as np
import pytest

from jeam.Models.HyperSpherical import HyperSphericalDiffusionModel
from jeam.Models.Spherical import SphericalDiffusionModel
from tests.numerical_diagnostics import (
    fixed_zero_drift_surface_density,
    trapezoid_mass,
    unit_sphere_surface_jacobian,
)

STANDARD_SPHERICAL_CASES = [
    pytest.param(
        3,
        SphericalDiffusionModel,
        np.array([[pi / 6, -0.8], [2 * pi / 3, 1.2]]),
        id="sdm",
    ),
    pytest.param(
        4,
        HyperSphericalDiffusionModel,
        np.array(
            [
                [pi / 6, pi / 4, -0.8],
                [2 * pi / 3, pi / 3, 1.2],
            ]
        ),
        id="hsdm",
    ),
]


@pytest.mark.parametrize(
    ("dimension", "model_type", "angles"), STANDARD_SPHERICAL_CASES
)
def test_standard_spherical_density_uses_exposed_angular_coordinates(
    dimension, model_type, angles
):
    decision_times = np.array([0.2, 0.8])
    observed = np.exp(
        model_type().joint_lpdf(
            rt=decision_times,
            theta=angles,
            drift_vec=np.zeros(dimension),
            ndt=0.0,
            threshold=1.0,
        )
    )
    surface_density = fixed_zero_drift_surface_density(dimension, decision_times)
    coordinate_jacobian = unit_sphere_surface_jacobian(*angles.T)

    np.testing.assert_allclose(
        observed,
        surface_density * coordinate_jacobian,
        rtol=2e-9,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("dimension", "model_type", "angles", "boundary_directions", "drift_vectors"),
    [
        pytest.param(
            3,
            SphericalDiffusionModel,
            np.array([[pi / 3, pi / 4], [2 * pi / 3, -pi / 6]]),
            np.array(
                [
                    [1 / 2, np.sqrt(6) / 4, np.sqrt(6) / 4],
                    [-1 / 2, 3 / 4, -np.sqrt(3) / 4],
                ]
            ),
            np.array([[0.45, -0.20, 0.35], [-0.15, 0.40, -0.25]]),
            id="sdm",
        ),
        pytest.param(
            4,
            HyperSphericalDiffusionModel,
            np.array(
                [
                    [pi / 3, pi / 4, pi / 6],
                    [2 * pi / 3, pi / 3, -pi / 4],
                ]
            ),
            np.array(
                [
                    [1 / 2, np.sqrt(6) / 4, 3 * np.sqrt(2) / 8, np.sqrt(6) / 8],
                    [-1 / 2, np.sqrt(3) / 4, 3 * np.sqrt(2) / 8, -3 * np.sqrt(2) / 8],
                ]
            ),
            np.array([[0.45, -0.20, 0.35, -0.10], [-0.15, 0.40, -0.25, 0.30]]),
            id="hsdm",
        ),
    ],
)
def test_standard_spherical_coordinate_density_with_nonzero_drift(
    dimension, model_type, angles, boundary_directions, drift_vectors
):
    decision_times = np.array([0.17, 0.63])
    threshold = 1.2
    sigma = 0.7

    observed = model_type().joint_lpdf(
        rt=decision_times,
        theta=angles,
        drift_vec=drift_vectors,
        ndt=0.0,
        threshold=threshold,
        sigma=sigma,
    )
    surface_density = fixed_zero_drift_surface_density(
        dimension,
        decision_times,
        threshold=threshold,
        sigma=sigma,
    )
    drift_projection = np.sum(drift_vectors * boundary_directions, axis=1)
    squared_drift_norm = np.sum(drift_vectors**2, axis=1)
    log_girsanov_factor = (
        threshold * drift_projection / sigma**2
        - 0.5 * squared_drift_norm * decision_times / sigma**2
    )
    expected = (
        np.log(surface_density)
        + log_girsanov_factor
        + np.log(unit_sphere_surface_jacobian(*angles.T))
    )

    np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    ("dimension", "model_type", "axes", "surface_area"),
    [
        pytest.param(
            3,
            SphericalDiffusionModel,
            (np.linspace(0.0, pi, 101), np.linspace(-pi, pi, 2)),
            4 * pi,
            id="sdm",
        ),
        pytest.param(
            4,
            HyperSphericalDiffusionModel,
            (
                np.linspace(0.0, pi, 101),
                np.linspace(0.0, pi, 101),
                np.linspace(-pi, pi, 2),
            ),
            2 * pi**2,
            id="hsdm",
        ),
    ],
)
def test_standard_spherical_density_integrates_over_complete_angular_domain(
    dimension, model_type, axes, surface_area
):
    decision_time = 0.5
    angular_grid = np.meshgrid(*axes, indexing="ij")
    angles = np.column_stack([coordinate.ravel() for coordinate in angular_grid])
    coordinate_density = np.exp(
        model_type().joint_lpdf(
            rt=np.full(angles.shape[0], decision_time),
            theta=angles,
            drift_vec=np.zeros(dimension),
            ndt=0.0,
            threshold=1.0,
        )
    ).reshape(angular_grid[0].shape)
    observed_angular_mass = trapezoid_mass(coordinate_density, axes)
    expected_angular_mass = (
        surface_area * fixed_zero_drift_surface_density(dimension, [decision_time])[0]
    )

    assert observed_angular_mass == pytest.approx(expected_angular_mass, rel=2e-4)


@pytest.mark.parametrize(
    ("dimension", "model_type", "polar_angles"),
    [
        pytest.param(
            3,
            SphericalDiffusionModel,
            np.array([[0.0, 0.0], [pi, 0.0]]),
            id="sdm",
        ),
        pytest.param(
            4,
            HyperSphericalDiffusionModel,
            np.array(
                [
                    [0.0, pi / 2, 0.0],
                    [pi, pi / 2, 0.0],
                    [pi / 2, 0.0, 0.0],
                ]
            ),
            id="hsdm",
        ),
    ],
)
def test_standard_spherical_coordinate_density_vanishes_at_poles(
    dimension, model_type, polar_angles
):
    log_density = model_type().joint_lpdf(
        rt=np.full(polar_angles.shape[0], 0.5),
        theta=polar_angles,
        drift_vec=np.zeros(dimension),
        ndt=0.0,
        threshold=1.0,
    )

    assert np.all(np.exp(log_density) < np.finfo(np.float64).eps)
