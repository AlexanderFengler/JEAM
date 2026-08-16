"""Characterize fixed CDM and PSDM likelihood input contracts."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from jeam.Models.Circular import CircularDiffusionModel
from jeam.Models.Spherical import ProjectedSphericalDiffusionModel

ModelType = type[CircularDiffusionModel] | type[ProjectedSphericalDiffusionModel]

FIXED_SINGLE_ANGLE_MODELS = [
    pytest.param(CircularDiffusionModel, np.array([-0.4, 0.7]), id="cdm"),
    pytest.param(ProjectedSphericalDiffusionModel, np.array([0.4, 2.1]), id="psdm"),
]


def _valid_arguments(theta: Any) -> dict[str, Any]:
    """Return one asymmetric, valid two-observation likelihood case."""
    return {
        "rt": np.array([0.35, 0.8]),
        "theta": theta,
        "drift_vec": np.array([0.45, 0.2]),
        "ndt": np.array([0.1, 0.15]),
        "threshold": 1.2,
        "s_v": 0.0,
        "s_t": 0.0,
        "sigma": 0.9,
        "approximation_step": 0.01,
    }


def _evaluate(
    model_type: ModelType,
    default_theta: Any,
    **overrides: Any,
) -> np.ndarray:
    """Evaluate a fixed likelihood after applying focused argument overrides."""
    arguments = _valid_arguments(default_theta)
    arguments.update(overrides)
    return model_type(threshold_dynamic="fixed").joint_lpdf(**arguments)


@pytest.mark.parametrize(("model_type", "theta"), FIXED_SINGLE_ANGLE_MODELS)
def test_fixed_likelihood_accepts_python_array_likes(model_type, theta):
    expected = _evaluate(model_type, theta)
    observed = _evaluate(
        model_type,
        theta.tolist(),
        rt=[0.35, 0.8],
        drift_vec=[0.45, 0.2],
        ndt=(0.1, 0.15),
    )

    assert observed.shape == (2,)
    assert observed.dtype == np.float64
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("model_type", "theta"),
    [
        pytest.param(CircularDiffusionModel, -0.4, id="cdm"),
        pytest.param(ProjectedSphericalDiffusionModel, 0.4, id="psdm"),
    ],
)
def test_fixed_likelihood_accepts_one_scalar_observation(model_type, theta):
    observed = _evaluate(
        model_type,
        theta,
        rt=0.35,
        drift_vec=(0.45, 0.2),
        ndt=0.1,
    )

    assert observed.shape == (1,)
    assert observed.dtype == np.float64
    assert np.isfinite(observed[0])


@pytest.mark.parametrize(("model_type", "theta"), FIXED_SINGLE_ANGLE_MODELS)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"rt": []}, "rt.*at least one", id="empty-rt"),
        pytest.param({"rt": [[0.35], [0.8]]}, "rt.*one-dimensional", id="matrix-rt"),
        pytest.param({"rt": [0.35, "bad"]}, "rt.*numeric", id="nonnumeric-rt"),
        pytest.param({"rt": [0.35, np.nan]}, "rt.*finite", id="nonfinite-rt"),
        pytest.param({"theta": [0.4]}, "theta.*same length", id="theta-length"),
        pytest.param(
            {"drift_vec": [[0.45, 0.2]]},
            "drift_vec.*one row per observation",
            id="drift-rows",
        ),
        pytest.param(
            {"drift_vec": [0.45, np.inf]}, "drift_vec.*finite", id="drift-finite"
        ),
        pytest.param({"ndt": [0.1]}, "ndt.*scalar or have length", id="ndt-length"),
        pytest.param({"ndt": [0.1, np.inf]}, "ndt.*finite", id="ndt-finite"),
    ],
)
def test_fixed_likelihood_rejects_malformed_observations(
    model_type,
    theta,
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _evaluate(model_type, theta, **overrides)


@pytest.mark.parametrize(("model_type", "theta"), FIXED_SINGLE_ANGLE_MODELS)
def test_fixed_likelihood_rejects_wrong_drift_dimension(model_type, theta):
    with pytest.raises(ValueError, match="drift_vec.*shape"):
        _evaluate(model_type, theta, drift_vec=np.ones((2, 3)))


@pytest.mark.parametrize(("model_type", "theta"), FIXED_SINGLE_ANGLE_MODELS)
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"ndt": -0.1}, "ndt.*non-negative", id="negative-ndt"),
        pytest.param({"threshold": 0.0}, "threshold.*positive", id="zero-threshold"),
        pytest.param(
            {"threshold": -1.0}, "threshold.*positive", id="negative-threshold"
        ),
        pytest.param(
            {"threshold": [1.0, 1.2]},
            "threshold.*scalar",
            id="vector-threshold",
        ),
        pytest.param({"s_v": -0.1}, "s_v.*non-negative", id="negative-s-v"),
        pytest.param({"s_t": -0.1}, "s_t.*non-negative", id="negative-s-t"),
        pytest.param({"sigma": 0.0}, "sigma.*positive", id="zero-sigma"),
        pytest.param({"sigma": -1.0}, "sigma.*positive", id="negative-sigma"),
        pytest.param(
            {"approximation_step": 0.0},
            "approximation_step.*positive",
            id="zero-approximation-step",
        ),
    ],
)
def test_fixed_likelihood_rejects_invalid_parameter_domains(
    model_type,
    theta,
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _evaluate(model_type, theta, **overrides)


def test_fixed_psdm_rejects_negative_projected_drift_component():
    with pytest.raises(ValueError, match="drift_vec.*second component.*non-negative"):
        _evaluate(
            ProjectedSphericalDiffusionModel,
            np.array([0.4, 2.1]),
            drift_vec=np.array([0.45, -0.2]),
        )


@pytest.mark.parametrize(("model_type", "theta"), FIXED_SINGLE_ANGLE_MODELS)
@pytest.mark.parametrize(
    ("parameter", "unexpected_function"),
    [
        pytest.param(
            "threshold_function",
            lambda time: np.ones_like(time),
            id="threshold-function",
        ),
        pytest.param(
            "dt_threshold_function",
            lambda time: np.zeros_like(time),
            id="threshold-derivative",
        ),
    ],
)
def test_fixed_likelihood_rejects_threshold_functions(
    model_type,
    theta,
    parameter: str,
    unexpected_function: Callable[[np.ndarray], np.ndarray],
):
    with pytest.raises(ValueError, match=f"{parameter}.*fixed"):
        _evaluate(model_type, theta, **{parameter: unexpected_function})
