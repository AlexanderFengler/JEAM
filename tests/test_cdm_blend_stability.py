"""Independent characterization of fixed-CDM blend cancellation."""

from math import log, pi

import mpmath as mp
import numpy as np
from scipy.special import jn_zeros, jv

from jeam.Models.Circular import CircularDiffusionModel
from jeam.utility.fpts import (
    cdm_long_t_fpt_z,
    cdm_long_t_fpt_z_roundoff_bound,
)


def _high_precision_log_fpt(scaled_time: float) -> float:
    """Evaluate the full zero-drift Bessel series beyond float64 precision."""
    with mp.workdps(120):
        roots = [mp.besseljzero(0, index) for index in range(1, 301)]
        terms = [
            root
            / mp.besselj(1, root)
            * mp.exp(-(root**2) * mp.mpf(str(scaled_time)) / 2)
            for root in roots
        ]
        density = mp.fsum(terms)
        return float(mp.log(density))


def _short_time_log_fpt(scaled_time: float) -> float:
    """Evaluate the published short-time approximation independently."""
    starting_radius_squared = 1e-8
    first_root = jn_zeros(0, 1)[0]
    return float(
        np.log1p(-starting_radius_squared)
        + 2.0 * np.log1p(scaled_time)
        - 0.5 * np.log(starting_radius_squared + scaled_time)
        - 1.5 * np.log(scaled_time)
        - 0.5 * (1.0 - starting_radius_squared) ** 2 / scaled_time
        - 0.5 * first_root**2 * scaled_time
    )


def _historical_blended_log_fpt(scaled_time: float) -> float:
    """Reconstruct a positive pre-fix residue without platform-dependent sign.

    Ordinary reductions leave a machine-scale residual whose sign depends on the
    reduction order. The historical defect occurs for the positive sign, so take the
    magnitude explicitly instead of making the regression architecture-dependent.
    """
    roots = jn_zeros(0, 100)
    coefficients = roots / jv(1, roots)
    terms = coefficients * np.exp(-(roots**2) * scaled_time / 2.0)
    long_density = abs(sum(float(term) for term in terms))
    blend_weight = np.clip((scaled_time - 0.002) / (0.02 - 0.002), 0.0, 1.0)
    short_density = np.exp(_short_time_log_fpt(scaled_time))
    blended_density = (1.0 - blend_weight) * short_density + blend_weight * long_density
    return float(np.log(blended_density))


def test_historical_fixed_cdm_blend_amplifies_cancellation_residue():
    """Positive roundoff residue should not masquerade as early-time density."""
    scaled_time = 0.003
    reference = _high_precision_log_fpt(scaled_time)
    short_time = _short_time_log_fpt(scaled_time)
    historical = _historical_blended_log_fpt(scaled_time)

    assert abs(short_time - reference) < 0.002
    assert historical - reference > 100.0
    assert historical > -50.0
    assert reference < -150.0


def test_fixed_cdm_blend_rejects_unreliable_positive_residue():
    """The corrected likelihood should follow the accurate early-time branch."""
    scaled_time = 0.003
    reference = _high_precision_log_fpt(scaled_time) - log(2.0 * pi)
    historical = _historical_blended_log_fpt(scaled_time) - log(2.0 * pi)
    observed = CircularDiffusionModel(threshold_dynamic="fixed").joint_lpdf(
        rt=np.array([scaled_time]),
        theta=np.array([0.0]),
        drift_vec=np.zeros(2),
        ndt=0.0,
        threshold=1.0,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )[0]

    assert historical - reference > 100.0
    np.testing.assert_allclose(
        observed,
        reference,
        rtol=0.0,
        atol=0.002,
    )


def test_cdm_roundoff_bound_separates_unreliable_and_resolved_series():
    """The guard should reject cancellation residue but retain resolved long values."""
    scaled_times = np.array([0.003, 0.018])
    long_density = cdm_long_t_fpt_z(scaled_times, threshold=1.0)
    roundoff_bound = cdm_long_t_fpt_z_roundoff_bound(
        scaled_times,
        threshold=1.0,
    )

    assert abs(long_density[0]) < roundoff_bound[0]
    assert long_density[1] > roundoff_bound[1] > 0.0


def test_corrected_cdm_blend_has_no_roundoff_spikes():
    """The rising early-time density should remain monotone through the guard switch."""
    scaled_times = np.linspace(0.012, 0.017, 5_000)
    observed = CircularDiffusionModel(threshold_dynamic="fixed").joint_lpdf(
        rt=scaled_times,
        theta=np.zeros_like(scaled_times),
        drift_vec=np.zeros(2),
        ndt=0.0,
        threshold=1.0,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )

    assert np.all(np.isfinite(observed))
    assert np.all(np.diff(observed) > 0.0)
