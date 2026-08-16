"""Input normalization shared by fixed single-angle likelihoods."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FixedSingleAngleInputs:
    """Validated, consistently shaped inputs for one-angle likelihoods."""

    rt: FloatArray
    theta: FloatArray
    drift_vec: FloatArray
    ndt: FloatArray
    threshold: float
    s_v: float
    s_t: float
    sigma: float
    approximation_step: float


def _numeric_array(value: Any, name: str) -> FloatArray:
    """Convert an array-like value without coercing non-numeric objects."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric.") from error

    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be numeric.")

    return np.asarray(array, dtype=np.float64)


def _observation_vector(value: Any, name: str) -> FloatArray:
    """Normalize one observation coordinate to a nonempty float vector."""
    array = _numeric_array(value, name)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim != 1:
        raise ValueError(
            f"{name} must be a scalar or one-dimensional numeric array."
        )

    if array.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array.copy()


def _drift_matrix(
    value: Any,
    n_observations: int,
    *,
    require_nonnegative_second_drift: bool,
) -> FloatArray:
    """Normalize a shared or trial-wise two-dimensional drift vector."""
    drift_vec = _numeric_array(value, "drift_vec")

    if drift_vec.ndim == 1:
        if drift_vec.shape != (2,):
            raise ValueError(
                "drift_vec must have shape (2,) or (n_observations, 2)."
            )
        drift_vec = np.broadcast_to(drift_vec, (n_observations, 2)).copy()
    elif drift_vec.ndim == 2 and drift_vec.shape[1] == 2:
        if drift_vec.shape[0] != n_observations:
            raise ValueError(
                "drift_vec must have one row per observation "
                f"({n_observations} rows)."
            )
        drift_vec = drift_vec.copy()
    else:
        raise ValueError(
            "drift_vec must have shape (2,) or (n_observations, 2)."
        )

    if not np.all(np.isfinite(drift_vec)):
        raise ValueError("drift_vec must contain only finite values.")
    if require_nonnegative_second_drift and np.any(drift_vec[:, 1] < 0):
        raise ValueError(
            "drift_vec second component must be non-negative for the "
            "projected spherical model."
        )

    return drift_vec


def _ndt_vector(value: Any, n_observations: int) -> FloatArray:
    """Normalize scalar or trial-wise non-decision times."""
    ndt = _numeric_array(value, "ndt")
    if ndt.ndim == 0:
        ndt = np.full(n_observations, float(ndt), dtype=np.float64)
    elif ndt.ndim == 1 and ndt.shape[0] == n_observations:
        ndt = ndt.copy()
    else:
        raise ValueError(
            f"ndt must be scalar or have length {n_observations}."
        )

    if not np.all(np.isfinite(ndt)):
        raise ValueError("ndt must contain only finite values.")
    if np.any(ndt < 0):
        raise ValueError("ndt must be non-negative.")

    return ndt


def _scalar(value: Any, name: str) -> float:
    """Return one finite numeric scalar."""
    scalar = _numeric_array(value, name)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a numeric scalar.")

    result = float(scalar)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_scalar(value: Any, name: str) -> float:
    """Return one finite, strictly positive numeric scalar."""
    result = _scalar(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_scalar(value: Any, name: str) -> float:
    """Return one finite, non-negative numeric scalar."""
    result = _scalar(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def normalize_fixed_single_angle_likelihood_inputs(
    *,
    rt: Any,
    theta: Any,
    drift_vec: Any,
    ndt: Any,
    threshold: Any,
    threshold_function: Any,
    dt_threshold_function: Any,
    s_v: Any,
    s_t: Any,
    sigma: Any,
    approximation_step: Any,
    require_nonnegative_second_drift: bool = False,
) -> FixedSingleAngleInputs:
    """Validate and normalize inputs used by fixed CDM and PSDM likelihoods.

    Response-support semantics are intentionally left to the likelihood. This
    helper owns only structural, finiteness, and parameter-domain validation.
    """
    if threshold_function is not None:
        raise ValueError("threshold_function must be None for a fixed threshold.")
    if dt_threshold_function is not None:
        raise ValueError(
            "dt_threshold_function must be None for a fixed threshold."
        )

    normalized_rt = _observation_vector(rt, "rt")
    normalized_theta = _observation_vector(theta, "theta")
    n_observations = normalized_rt.shape[0]
    if normalized_theta.shape[0] != n_observations:
        raise ValueError("theta must have the same length as rt.")

    return FixedSingleAngleInputs(
        rt=normalized_rt,
        theta=normalized_theta,
        drift_vec=_drift_matrix(
            drift_vec,
            n_observations,
            require_nonnegative_second_drift=(
                require_nonnegative_second_drift
            ),
        ),
        ndt=_ndt_vector(ndt, n_observations),
        threshold=_positive_scalar(threshold, "threshold"),
        s_v=_nonnegative_scalar(s_v, "s_v"),
        s_t=_nonnegative_scalar(s_t, "s_t"),
        sigma=_positive_scalar(sigma, "sigma"),
        approximation_step=_positive_scalar(
            approximation_step, "approximation_step"
        ),
    )
