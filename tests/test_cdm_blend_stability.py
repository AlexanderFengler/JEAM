"""Independent characterization of fixed-CDM blend cancellation."""

import mpmath as mp
import numpy as np
from scipy.special import jn_zeros, jv


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
    """Reconstruct the pre-fix float64 blend without importing JEAM."""
    roots = jn_zeros(0, 100)
    coefficients = roots / jv(1, roots)
    long_density = np.sum(coefficients * np.exp(-(roots**2) * scaled_time / 2.0))
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
