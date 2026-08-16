"""Differentiable fixed-boundary likelihoods implemented with JAX.

The fixed CDM likelihood is differentiable almost everywhere. Its response-support
boundaries and the two short/long-time blend knots are intentionally non-smooth.
"""

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from jeam.utility.Constants import JVZ1, zeros_0

_SHORT_TIME_END = 0.002
_LONG_TIME_START = 0.02
_STARTING_RADIUS_SQUARED = 0.1**8
_LOG_TWO_PI = float(np.log(2.0 * np.pi))
_ZEROS_0 = np.asarray(zeros_0, dtype=np.float64)
_LONG_TIME_COEFFICIENTS = np.asarray(zeros_0 / JVZ1, dtype=np.float64)


def _contains_tracer(values: Sequence[Any]) -> bool:
    """Return whether JAX is tracing any value in an eager wrapper call."""
    return any(isinstance(value, jax.core.Tracer) for value in values)


def _numeric_vector(value: Any, name: str) -> np.ndarray:
    """Normalize a concrete scalar or vector for eager validation."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error

    if array.dtype.kind not in "iuf" or array.ndim > 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty numeric scalar or vector.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _validate_concrete_inputs(
    rt: Any,
    theta: Any,
    v_x: Any,
    v_y: Any,
    threshold: Any,
    ndt: Any,
    sigma: Any,
    log_density_floor: Any,
) -> None:
    """Raise targeted input errors outside traced likelihood evaluation."""
    values = (rt, theta, v_x, v_y, threshold, ndt, sigma)
    traced_values = values + (log_density_floor,)
    if _contains_tracer(traced_values):
        return

    names = ("rt", "theta", "v_x", "v_y", "a", "t", "sigma")
    arrays = tuple(
        _numeric_vector(value, name) for value, name in zip(values, names, strict=True)
    )
    try:
        np.broadcast_shapes(*(array.shape for array in arrays))
    except ValueError as error:
        raise ValueError(
            "rt, theta, v_x, v_y, a, t, and sigma must be broadcastable to "
            "one trial shape."
        ) from error

    threshold_array = arrays[4]
    ndt_array = arrays[5]
    sigma_array = arrays[6]
    if np.any(threshold_array <= 0):
        raise ValueError("a must be positive.")
    if np.any(ndt_array < 0):
        raise ValueError("t must be non-negative.")
    if np.any(sigma_array <= 0):
        raise ValueError("sigma must be positive.")

    if log_density_floor is not None:
        floor = np.asarray(log_density_floor)
        if floor.ndim != 0 or floor.dtype.kind not in "iuf" or not np.isfinite(floor):
            raise ValueError("log_density_floor must be a finite numeric scalar.")


def _short_time_log_fpt(
    decision_time: jax.Array,
    threshold: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Evaluate the zero-drift short-time approximation in log space."""
    sigma_squared = sigma**2
    threshold_squared = threshold**2
    scaled_time = sigma_squared * decision_time / threshold_squared
    scaled_radius = sigma_squared * _STARTING_RADIUS_SQUARED / threshold_squared
    first_zero = jnp.asarray(_ZEROS_0[0], dtype=decision_time.dtype)
    return (
        jnp.log(sigma_squared / threshold_squared)
        + jnp.log1p(-scaled_radius)
        + 2.0 * jnp.log1p(scaled_time)
        - 0.5 * jnp.log(scaled_radius + scaled_time)
        - 1.5 * jnp.log(scaled_time)
        - 0.5 * (1.0 - scaled_radius) ** 2 / scaled_time
        - 0.5 * first_zero**2 * scaled_time
    )


def _long_time_log_fpt(
    decision_time: jax.Array,
    threshold: jax.Array,
    sigma: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return log-absolute density and sign for the alternating Bessel series.

    Terms are exponentially scaled and accumulated in their declared order with Kahan
    compensation. This avoids both exponent underflow and reduction-tree-dependent
    cancellation in the short/long-time overlap.
    """
    dtype = decision_time.dtype
    roots = jnp.asarray(_ZEROS_0, dtype=dtype)
    coefficients = jnp.asarray(_LONG_TIME_COEFFICIENTS, dtype=dtype)
    scale = sigma**2 / threshold**2
    log_terms = (
        jnp.log(jnp.abs(coefficients))
        - (roots**2 * sigma**2) / (2.0 * threshold**2) * decision_time
    )
    maximum_log_term = jnp.max(log_terms)
    scaled_terms = jnp.sign(coefficients) * jnp.exp(log_terms - maximum_log_term)

    def compensated_add(
        carry: tuple[jax.Array, jax.Array],
        term: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        total, compensation = carry
        corrected_term = term - compensation
        updated_total = total + corrected_term
        updated_compensation = (updated_total - total) - corrected_term
        return (updated_total, updated_compensation), None

    initial = (jnp.zeros((), dtype=dtype), jnp.zeros((), dtype=dtype))
    (scaled_series, _), _ = jax.lax.scan(
        compensated_add,
        initial,
        scaled_terms,
    )
    sign = jnp.sign(scaled_series)
    log_absolute_series = maximum_log_term + jnp.log(jnp.abs(scaled_series))
    return jnp.log(scale) + log_absolute_series, sign


def fixed_cdm_logpdf_single(
    rt: jax.Array,
    theta: jax.Array,
    v_x: jax.Array,
    v_y: jax.Array,
    a: jax.Array,
    t: jax.Array,
    sigma: jax.Array = 1.0,
) -> jax.Array:
    """Evaluate one strict fixed-CDM log density with a pure traced kernel.

    Infeasible response or parameter proposals return negative infinity. A non-finite
    result on otherwise valid support remains visible as a numerical failure.
    """
    result_dtype = jnp.result_type(rt, theta, v_x, v_y, a, t, sigma, jnp.float32)
    # The alternating Bessel series is ill-conditioned in the overlap region.
    # Evaluate it at the recovery environment's required x64 precision even when a
    # caller requests float32 output, then cast only the final pointwise value.
    dtype = jnp.dtype(jnp.float64) if jax.config.x64_enabled else result_dtype
    rt, theta, v_x, v_y, a, t, sigma = (
        jnp.asarray(value, dtype=dtype) for value in (rt, theta, v_x, v_y, a, t, sigma)
    )
    finite_inputs = jnp.all(
        jnp.stack(
            tuple(jnp.isfinite(value) for value in (rt, theta, v_x, v_y, a, t, sigma))
        )
    )
    parameter_support = (a > 0.0) & (t >= 0.0) & (sigma > 0.0)
    response_support = (rt > t) & (theta >= -jnp.pi) & (theta < jnp.pi)
    support = finite_inputs & parameter_support & response_support

    safe_decision_time = jnp.where(rt > t, rt - t, jnp.asarray(1.0, dtype=dtype))
    safe_threshold = jnp.where(a > 0.0, a, jnp.asarray(1.0, dtype=dtype))
    safe_sigma = jnp.where(sigma > 0.0, sigma, jnp.asarray(1.0, dtype=dtype))

    scaled_time = safe_decision_time / safe_threshold**2
    blend_weight = jnp.clip(
        (scaled_time - _SHORT_TIME_END) / (_LONG_TIME_START - _SHORT_TIME_END),
        0.0,
        1.0,
    )
    short_log_density = _short_time_log_fpt(
        safe_decision_time,
        safe_threshold,
        safe_sigma,
    )
    long_log_absolute_density, long_sign = _long_time_log_fpt(
        safe_decision_time,
        safe_threshold,
        safe_sigma,
    )

    def evaluate_blend(_: None) -> jax.Array:
        blend_logs = jnp.stack(
            (
                jnp.log1p(-blend_weight) + short_log_density,
                jnp.log(blend_weight) + long_log_absolute_density,
            )
        )
        blended_log_absolute_density, blended_sign = logsumexp(
            blend_logs,
            b=jnp.stack((jnp.ones_like(long_sign), long_sign)),
            return_sign=True,
        )
        return jnp.where(
            blended_sign > 0.0,
            blended_log_absolute_density,
            short_log_density,
        )

    def evaluate_nonshort(_: None) -> jax.Array:
        return jax.lax.cond(
            blend_weight >= 1.0,
            lambda unused: jnp.where(
                long_sign > 0.0,
                long_log_absolute_density,
                jnp.nan,
            ),
            evaluate_blend,
            operand=None,
        )

    fpt_log_density = jax.lax.cond(
        blend_weight <= 0.0,
        lambda unused: short_log_density,
        evaluate_nonshort,
        operand=None,
    )

    drift_projection = v_x * jnp.cos(theta) + v_y * jnp.sin(theta)
    log_density = (
        safe_threshold * drift_projection / safe_sigma**2
        - 0.5 * (v_x**2 + v_y**2) * safe_decision_time / safe_sigma**2
        + fpt_log_density
        - jnp.asarray(_LOG_TWO_PI, dtype=dtype)
    )
    return jnp.asarray(
        jnp.where(support, log_density, -jnp.inf),
        dtype=result_dtype,
    )


def fixed_cdm_logpdf(
    rt: Any,
    theta: Any,
    v_x: Any,
    v_y: Any,
    a: Any,
    t: Any,
    *,
    sigma: Any = 1.0,
    log_density_floor: Any = None,
) -> jax.Array:
    """Evaluate the fixed-CDM log density for broadcastable trial inputs.

    Concrete calls validate finite values, shapes, and parameter domains eagerly. During
    JAX tracing, invalid proposals are handled by :func:`fixed_cdm_logpdf_single` and
    return negative infinity. An explicit ``log_density_floor`` is applied only after the
    strict result and therefore remains an opt-in algorithmic approximation.
    """
    _validate_concrete_inputs(
        rt,
        theta,
        v_x,
        v_y,
        a,
        t,
        sigma,
        log_density_floor,
    )
    arrays = jnp.broadcast_arrays(
        *(
            jnp.atleast_1d(jnp.asarray(value))
            for value in (rt, theta, v_x, v_y, a, t, sigma)
        )
    )
    log_density = jax.vmap(fixed_cdm_logpdf_single)(*arrays)
    if log_density_floor is not None:
        log_density = jnp.maximum(log_density, jnp.asarray(log_density_floor))
    return log_density


__all__ = ["fixed_cdm_logpdf", "fixed_cdm_logpdf_single"]
