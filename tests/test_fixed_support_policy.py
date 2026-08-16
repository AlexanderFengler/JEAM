"""Characterize fixed CDM and PSDM support and numerical-failure policy."""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import jeam.Models.Circular as circular_module
import jeam.Models.Spherical as spherical_module
from jeam.Models.Circular import CircularDiffusionModel
from jeam.Models.Spherical import ProjectedSphericalDiffusionModel

ModelType = type[CircularDiffusionModel] | type[ProjectedSphericalDiffusionModel]

FIXED_MODELS = [
    pytest.param(CircularDiffusionModel, np.array([-0.4, 0.7]), id="cdm"),
    pytest.param(ProjectedSphericalDiffusionModel, np.array([0.4, 2.1]), id="psdm"),
]

FPT_IMPLEMENTATIONS = [
    pytest.param(
        CircularDiffusionModel,
        np.array([-0.4, 0.7]),
        circular_module,
        ("cdm_short_t_fpt_z", "cdm_long_t_fpt_z"),
        id="cdm",
    ),
    pytest.param(
        ProjectedSphericalDiffusionModel,
        np.array([0.4, 2.1]),
        spherical_module,
        ("sdm_short_t_fpt_z", "sdm_long_t_fpt_z"),
        id="psdm",
    ),
]

SUPPORT_POLICY_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="Fixed likelihoods still floor support and numerical failures.",
)


def _arguments(theta: Any) -> dict[str, Any]:
    """Return one valid, asymmetric fixed-likelihood argument set."""
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
    arguments = _arguments(default_theta)
    arguments.update(overrides)
    return model_type(threshold_dynamic="fixed").joint_lpdf(**arguments)


@SUPPORT_POLICY_XFAIL
@pytest.mark.parametrize(("model_type", "theta"), FIXED_MODELS)
def test_fixed_likelihood_returns_negative_infinity_outside_rt_support(
    model_type,
    theta,
):
    observed = _evaluate(
        model_type,
        theta,
        rt=np.array([0.1, 0.05]),
        ndt=np.array([0.1, 0.1]),
    )

    assert np.all(np.isneginf(observed))


@SUPPORT_POLICY_XFAIL
@pytest.mark.parametrize(("model_type", "theta"), FIXED_MODELS)
def test_fixed_likelihood_preserves_mixed_rt_support(model_type, theta):
    observed = _evaluate(
        model_type,
        theta,
        rt=np.array([0.35, 0.1]),
        ndt=np.array([0.1, 0.1]),
    )

    assert np.isfinite(observed[0])
    assert np.isneginf(observed[1])


@SUPPORT_POLICY_XFAIL
def test_fixed_cdm_enforces_half_open_angle_support():
    theta = np.array([-np.pi, np.pi, -np.pi - 1e-9, np.pi + 1e-9])
    observed = _evaluate(
        CircularDiffusionModel,
        theta,
        rt=np.full(theta.shape, 0.8),
        ndt=0.1,
    )

    assert np.isfinite(observed[0])
    assert np.all(np.isneginf(observed[1:]))


@SUPPORT_POLICY_XFAIL
def test_fixed_psdm_enforces_closed_angle_domain_and_zero_density_at_poles():
    theta = np.array([0.0, np.pi, -1e-9, np.pi + 1e-9, np.pi / 2])
    observed = _evaluate(
        ProjectedSphericalDiffusionModel,
        theta,
        rt=np.full(theta.shape, 0.8),
        ndt=0.1,
    )

    assert np.all(np.isneginf(observed[:4]))
    assert np.isfinite(observed[4])


def _constant_fpt(value: float) -> Callable[..., np.ndarray]:
    def evaluate(time, *args, **kwargs):
        del args, kwargs
        return np.full_like(np.asarray(time, dtype=np.float64), value)

    return evaluate


@SUPPORT_POLICY_XFAIL
@pytest.mark.parametrize(
    ("model_type", "theta", "module", "function_names"), FPT_IMPLEMENTATIONS
)
@pytest.mark.parametrize("failed_value", [np.nan, -1.0], ids=["nonfinite", "negative"])
def test_fixed_likelihood_raises_for_failed_first_passage_computation(
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
    theta: np.ndarray,
    module: ModuleType,
    function_names: tuple[str, str],
    failed_value: float,
):
    for function_name in function_names:
        monkeypatch.setattr(module, function_name, _constant_fpt(failed_value))

    with pytest.raises(FloatingPointError, match="first-passage density"):
        _evaluate(model_type, theta)


@SUPPORT_POLICY_XFAIL
@pytest.mark.parametrize(("model_type", "theta"), FIXED_MODELS)
def test_fixed_likelihood_raises_for_nonfinite_joint_density(model_type, theta):
    with pytest.raises(FloatingPointError, match="joint density"):
        _evaluate(model_type, theta, drift_vec=np.array([1e308, 1e308]))
