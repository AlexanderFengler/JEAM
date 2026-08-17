"""Scientific consistency gates for the fixed projected spherical model."""

from math import pi

import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.special import iv

from jeam.Models.Spherical import ProjectedSphericalDiffusionModel
from tests.numerical_diagnostics import fixed_psdm_coordinate_density


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
