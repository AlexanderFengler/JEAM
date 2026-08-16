"""Characterize fixed CDM and PSDM support and numerical-failure policy."""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from scipy.special import jn_zeros

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
        "cdm_long_t_fpt_z",
        "cdm_short_t_log_fpt_z",
        id="cdm",
    ),
    pytest.param(
        ProjectedSphericalDiffusionModel,
        np.array([0.4, 2.1]),
        spherical_module,
        "sdm_long_t_fpt_z",
        "sdm_short_t_log_fpt_z",
        id="psdm",
    ),
]

LEGACY_NDT_VARIABILITY_VALUES = [
    pytest.param(
        CircularDiffusionModel,
        np.array([-0.4, 0.7]),
        np.array([-1.845558254119741, -1.336365150184658]),
        id="cdm",
    ),
    pytest.param(
        ProjectedSphericalDiffusionModel,
        np.array([0.4, 2.1]),
        np.array([-0.795517992922905, -1.2952326109538082]),
        id="psdm",
    ),
]

LOG_DENSITY_FLOOR = -66.1


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


@pytest.mark.xfail(
    strict=True,
    reason="Optional fixed-likelihood floors are not implemented yet.",
)
@pytest.mark.parametrize(("model_type", "theta"), FIXED_MODELS)
def test_fixed_likelihood_applies_opt_in_floor_after_strict_evaluation(
    model_type,
    theta,
):
    """A floor should cover support and tail values without changing the interior."""
    response = np.full(3, theta[0])
    rt = np.array([0.1, 0.100001, 0.8])
    strict = _evaluate(
        model_type,
        response,
        rt=rt,
        ndt=0.1,
    )
    floored = _evaluate(
        model_type,
        response,
        rt=rt,
        ndt=0.1,
        log_density_floor=LOG_DENSITY_FLOOR,
    )

    assert np.isneginf(strict[0])
    assert strict[1] < LOG_DENSITY_FLOOR
    assert strict[2] > LOG_DENSITY_FLOOR
    np.testing.assert_allclose(
        floored,
        np.maximum(strict, LOG_DENSITY_FLOOR),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.xfail(
    strict=True,
    reason="Optional fixed-likelihood floors are not implemented yet.",
)
@pytest.mark.parametrize(("model_type", "theta"), FIXED_MODELS)
@pytest.mark.parametrize(
    "floor",
    [np.nan, np.inf, -np.inf, [-66.1]],
    ids=["nan", "positive-infinity", "negative-infinity", "array"],
)
def test_fixed_likelihood_rejects_invalid_log_density_floor(
    model_type,
    theta,
    floor,
):
    with pytest.raises(ValueError, match="log_density_floor"):
        _evaluate(model_type, theta, log_density_floor=floor)


def _constant_fpt(value: float) -> Callable[..., np.ndarray]:
    def evaluate(time, *args, **kwargs):
        del args, kwargs
        return np.full_like(np.asarray(time, dtype=np.float64), value)

    return evaluate


@pytest.mark.parametrize(
    ("model_type", "theta", "module", "long_function_name", "short_function_name"),
    FPT_IMPLEMENTATIONS,
)
@pytest.mark.parametrize("failed_value", [np.nan, -1.0], ids=["nonfinite", "negative"])
def test_fixed_likelihood_raises_for_failed_first_passage_computation(
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
    theta: np.ndarray,
    module: ModuleType,
    long_function_name: str,
    short_function_name: str,
    failed_value: float,
):
    del short_function_name
    monkeypatch.setattr(module, long_function_name, _constant_fpt(failed_value))

    with pytest.raises(FloatingPointError, match="first-passage density"):
        _evaluate(model_type, theta)


@pytest.mark.xfail(
    strict=True,
    reason="Optional fixed-likelihood floors are not implemented yet.",
)
@pytest.mark.parametrize(
    ("model_type", "theta", "module", "long_function_name", "short_function_name"),
    FPT_IMPLEMENTATIONS,
)
def test_fixed_likelihood_floor_does_not_mask_numerical_failure(
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
    theta: np.ndarray,
    module: ModuleType,
    long_function_name: str,
    short_function_name: str,
):
    del short_function_name
    monkeypatch.setattr(module, long_function_name, _constant_fpt(np.nan))

    with pytest.raises(FloatingPointError, match="first-passage density"):
        _evaluate(
            model_type,
            theta,
            log_density_floor=LOG_DENSITY_FLOOR,
        )


@pytest.mark.parametrize(
    ("model_type", "theta", "module", "long_function_name", "short_function_name"),
    FPT_IMPLEMENTATIONS,
)
def test_fixed_likelihood_raises_for_nonfinite_blended_fpt(
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
    theta: np.ndarray,
    module: ModuleType,
    long_function_name: str,
    short_function_name: str,
):
    del short_function_name
    monkeypatch.setattr(module, long_function_name, _constant_fpt(np.nan))

    with pytest.raises(FloatingPointError, match="first-passage density"):
        _evaluate(model_type, theta, rt=np.array([0.1144, 0.1144]), ndt=0.1)


@pytest.mark.parametrize(
    ("model_type", "theta", "module", "long_function_name", "short_function_name"),
    FPT_IMPLEMENTATIONS,
)
def test_fixed_likelihood_raises_for_nonfinite_short_time_fpt(
    monkeypatch: pytest.MonkeyPatch,
    model_type: ModelType,
    theta: np.ndarray,
    module: ModuleType,
    long_function_name: str,
    short_function_name: str,
):
    del long_function_name
    monkeypatch.setattr(module, short_function_name, _constant_fpt(np.nan))

    with pytest.raises(FloatingPointError, match="first-passage density"):
        _evaluate(model_type, theta, rt=np.array([0.101, 0.101]), ndt=0.1)


@pytest.mark.parametrize(("model_type", "theta"), FIXED_MODELS)
def test_fixed_likelihood_raises_for_nonfinite_joint_density(model_type, theta):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        with pytest.raises(FloatingPointError, match="joint density"):
            _evaluate(model_type, theta, drift_vec=np.array([1e308, 1e308]))


@pytest.mark.parametrize(
    ("model_type", "theta", "dimension"),
    [
        pytest.param(CircularDiffusionModel, -0.4, 2, id="cdm"),
        pytest.param(ProjectedSphericalDiffusionModel, 0.4, 3, id="psdm"),
    ],
)
def test_fixed_likelihood_preserves_short_time_log_density(
    model_type,
    theta,
    dimension,
):
    ndt = 0.1
    rt = ndt + 1e-6
    threshold = 1.2
    sigma = 0.9
    observed = _evaluate(
        model_type,
        theta,
        rt=rt,
        ndt=ndt,
        threshold=threshold,
        sigma=sigma,
        drift_vec=np.zeros(2),
    )

    scaled_time = sigma**2 * (rt - ndt) / threshold**2
    start = sigma**2 * 1e-8 / threshold**2
    if dimension == 2:
        short_log_density = (
            np.log1p(-start)
            + 2 * np.log1p(scaled_time)
            - 0.5 * np.log(start + scaled_time)
            - 1.5 * np.log(scaled_time)
            - 0.5 * (1 - start) ** 2 / scaled_time
            - 0.5 * jn_zeros(0, 1)[0] ** 2 * scaled_time
        )
        response_log_density = -np.log(2 * np.pi)
    else:
        short_log_density = (
            np.log1p(-start)
            + 2.5 * np.log1p(scaled_time)
            - np.log(start + scaled_time)
            - 1.5 * np.log(scaled_time)
            - 0.5 * (1 - start) ** 2 / scaled_time
            - 0.5 * np.pi**2 * scaled_time
        )
        response_log_density = -0.5 * np.log(2 * np.pi) + np.log(np.sin(theta))
    expected = (
        np.log(sigma**2 / threshold**2) + short_log_density + response_log_density
    )

    assert np.isfinite(observed[0])
    np.testing.assert_allclose(observed[0], expected, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(
    ("model_type", "theta", "expected"), LEGACY_NDT_VARIABILITY_VALUES
)
def test_fixed_support_policy_does_not_change_ndt_variability_branch(
    model_type,
    theta,
    expected,
):
    """NDT convolution support remains isolated to its dedicated follow-up."""
    observed = _evaluate(model_type, theta, s_t=0.05)

    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)
