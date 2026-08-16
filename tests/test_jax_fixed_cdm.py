"""Value and gradient contract for the fixed-CDM JAX likelihood."""

import subprocess
import sys
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from jeam.likelihoods.jax_fixed import fixed_cdm_logpdf, fixed_cdm_logpdf_single
from jeam.Models.Circular import CircularDiffusionModel

jax.config.update("jax_enable_x64", True)

FloatArray = NDArray[np.float64]
PARAMETER_NAMES = ("v_x", "v_y", "a", "t")
FINITE_DIFFERENCE_STEP = 1e-5


@dataclass(frozen=True)
class FixedCDMCase:
    """One protected point away from support and blend boundaries."""

    name: str
    rt: float
    theta: float
    parameters: tuple[float, float, float, float]
    scaled_decision_time: float
    expected_logpdf: float
    expected_gradient: tuple[float, float, float, float]


FIXED_CDM_CASES = (
    FixedCDMCase(
        name="short-time",
        rt=0.0812,
        theta=-2.1,
        parameters=(0.45, -0.30, 0.80, 0.08),
        scaled_decision_time=0.001875,
        expected_logpdf=-255.4763843380079,
        expected_gradient=(
            -0.4044168832706418,
            -0.6902074929371337,
            -664.1306876773001,
            -220569.4014592012,
        ),
    ),
    FixedCDMCase(
        name="blend",
        rt=0.10843136,
        theta=0.35,
        parameters=(-0.20, 0.65, 1.20, 0.08),
        scaled_decision_time=0.019744,
        expected_logpdf=-19.65145811995678,
        expected_gradient=(
            1.132933527436819,
            0.3929969849991721,
            -40.50537196009785,
            -820.0886038892462,
        ),
    ),
    FixedCDMCase(
        name="long-time",
        rt=0.442,
        theta=-1.25,
        parameters=(0.35, -0.55, 1.10, 0.20),
        scaled_decision_time=0.20,
        expected_logpdf=-0.7717810052178282,
        expected_gradient=(
            0.262154598629216,
            -0.910783081276012,
            -1.887927976365863,
            -1.383069389671032,
        ),
    ),
)


def _numpy_fixed_cdm_logpdf(
    case: FixedCDMCase,
    parameters: FloatArray,
) -> float:
    """Evaluate the corrected NumPy implementation without a JAX dependency."""
    v_x, v_y, threshold, ndt = parameters
    model = CircularDiffusionModel(threshold_dynamic="fixed")
    result = model.joint_lpdf(
        rt=np.array([case.rt]),
        theta=np.array([case.theta]),
        drift_vec=np.array([[v_x, v_y]]),
        ndt=np.array([ndt]),
        threshold=threshold,
        decay=0.0,
        threshold_function=None,
        dt_threshold_function=None,
        s_v=0.0,
        s_t=0.0,
        sigma=1.0,
    )
    return float(result[0])


def _central_difference_gradient(
    case: FixedCDMCase,
    parameters: FloatArray,
) -> FloatArray:
    """Differentiate the NumPy oracle independently with central differences."""
    gradient = np.empty(parameters.shape, dtype=np.float64)
    for index, value in enumerate(parameters):
        step = FINITE_DIFFERENCE_STEP * max(1.0, abs(value))
        above = parameters.copy()
        below = parameters.copy()
        above[index] += step
        below[index] -= step
        gradient[index] = (
            _numpy_fixed_cdm_logpdf(case, above) - _numpy_fixed_cdm_logpdf(case, below)
        ) / (2.0 * step)
    return gradient


@pytest.mark.parametrize("case", FIXED_CDM_CASES, ids=lambda case: case.name)
def test_numpy_oracle_protects_fixed_cdm_regimes(case):
    """Lock the corrected reference values before adding the JAX implementation."""
    parameters = np.asarray(case.parameters, dtype=np.float64)
    scaled_time = (case.rt - parameters[3]) / parameters[2] ** 2
    observed_logpdf = _numpy_fixed_cdm_logpdf(case, parameters)
    observed_gradient = _central_difference_gradient(case, parameters)

    assert scaled_time == pytest.approx(case.scaled_decision_time, abs=1e-15)
    assert observed_logpdf == pytest.approx(case.expected_logpdf, abs=5e-7)
    np.testing.assert_allclose(
        observed_gradient,
        np.asarray(case.expected_gradient),
        rtol=5e-4,
        atol=5e-7,
    )


def _case_arrays(dtype: np.dtype) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return observation and parameter arrays in protected row order."""
    response_times = np.asarray([case.rt for case in FIXED_CDM_CASES], dtype=dtype)
    angles = np.asarray([case.theta for case in FIXED_CDM_CASES], dtype=dtype)
    parameters = np.asarray(
        [case.parameters for case in FIXED_CDM_CASES],
        dtype=dtype,
    )
    return response_times, angles, parameters


def _numpy_case_values(parameters: FloatArray) -> FloatArray:
    """Evaluate the independent NumPy oracle in protected row order."""
    return np.asarray(
        [
            _numpy_fixed_cdm_logpdf(case, row.astype(np.float64))
            for case, row in zip(FIXED_CDM_CASES, parameters, strict=True)
        ]
    )


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        pytest.param(np.float64, 5e-10, 5e-12, id="float64"),
        pytest.param(np.float32, 2e-5, 2e-6, id="float32"),
    ],
)
def test_jax_fixed_cdm_matches_numpy_values(dtype, rtol, atol):
    """JAX should preserve protected values in every approximation regime."""
    response_times, angles, parameters = _case_arrays(dtype)
    observed = fixed_cdm_logpdf(
        response_times,
        angles,
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 2],
        parameters[:, 3],
    )
    expected = _numpy_case_values(parameters)

    assert observed.dtype == jnp.dtype(dtype)
    np.testing.assert_allclose(observed, expected, rtol=rtol, atol=atol)


def test_jax_fixed_cdm_gradients_match_independent_finite_differences():
    """Autodiff should match central differences away from knots and support."""
    response_times, angles, parameters = _case_arrays(np.float64)

    def row_logpdf(
        row_parameters: jax.Array,
        response_time: jax.Array,
        angle: jax.Array,
    ) -> jax.Array:
        return fixed_cdm_logpdf(
            response_time,
            angle,
            row_parameters[0],
            row_parameters[1],
            row_parameters[2],
            row_parameters[3],
        )[0]

    observed = jax.vmap(jax.grad(row_logpdf))(
        jnp.asarray(parameters),
        jnp.asarray(response_times),
        jnp.asarray(angles),
    )
    expected = np.asarray(
        [
            _central_difference_gradient(case, row)
            for case, row in zip(FIXED_CDM_CASES, parameters, strict=True)
        ]
    )

    assert np.all(np.isfinite(observed))
    np.testing.assert_allclose(observed, expected, rtol=2e-4, atol=1e-5)


def test_jax_fixed_cdm_broadcasts_scalars_and_preserves_trialwise_order():
    """Scalar parameters and trial-wise parameters should share one row contract."""
    response_times = np.array([0.31, 0.52, 0.84])
    angles = np.array([-1.2, 0.4, 2.1])
    scalar_result = fixed_cdm_logpdf(
        response_times,
        angles,
        0.45,
        -0.30,
        1.20,
        0.08,
    )
    trialwise_result = fixed_cdm_logpdf(
        response_times,
        angles,
        np.full(3, 0.45),
        np.full(3, -0.30),
        np.full(3, 1.20),
        np.full(3, 0.08),
    )
    permutation = np.array([2, 0, 1])
    permuted_result = fixed_cdm_logpdf(
        response_times[permutation],
        angles[permutation],
        0.45,
        -0.30,
        1.20,
        0.08,
    )

    assert scalar_result.shape == (3,)
    np.testing.assert_array_equal(trialwise_result, scalar_result)
    np.testing.assert_array_equal(permuted_result, scalar_result[permutation])


def test_jax_fixed_cdm_jit_reuses_a_same_shape_trace():
    """A second same-shaped call should reuse the first compiled trace."""
    trace_count = 0

    def traceable_logpdf(response_times, angles, v_x):
        nonlocal trace_count
        trace_count += 1
        return fixed_cdm_logpdf(
            response_times,
            angles,
            v_x,
            -0.25,
            1.15,
            0.07,
        )

    compiled_logpdf = jax.jit(traceable_logpdf)
    first = compiled_logpdf(
        jnp.array([0.28, 0.61]),
        jnp.array([-0.8, 1.4]),
        jnp.array([0.30, -0.10]),
    )
    second = compiled_logpdf(
        jnp.array([0.35, 0.72]),
        jnp.array([0.2, -1.1]),
        jnp.array([0.15, 0.40]),
    )

    assert trace_count == 1
    assert first.shape == second.shape == (2,)
    assert np.all(np.isfinite(first))
    assert np.all(np.isfinite(second))


def test_jax_fixed_cdm_preserves_strict_support_and_opt_in_floor():
    """The public default should be strict while an explicit floor remains available."""
    response_times = np.array([0.08, 0.31, 0.52])
    angles = np.array([0.2, np.pi, -0.7])
    strict = fixed_cdm_logpdf(
        response_times,
        angles,
        0.45,
        -0.30,
        1.20,
        0.08,
    )
    floored = fixed_cdm_logpdf(
        response_times,
        angles,
        0.45,
        -0.30,
        1.20,
        0.08,
        log_density_floor=-66.1,
    )

    assert np.all(np.isneginf(strict[:2]))
    assert np.isfinite(strict[2])
    np.testing.assert_array_equal(floored, jnp.maximum(strict, -66.1))


def test_jax_fixed_cdm_traced_invalid_proposals_return_negative_infinity():
    """Sampler proposals should use kernel support rather than eager exceptions."""
    traced_logpdf = jax.jit(
        lambda threshold, ndt: fixed_cdm_logpdf(
            0.40,
            0.20,
            0.45,
            -0.30,
            threshold,
            ndt,
        )
    )

    assert np.isneginf(traced_logpdf(-1.0, 0.08)[0])
    assert np.isneginf(traced_logpdf(1.20, -0.08)[0])
    assert np.isneginf(fixed_cdm_logpdf_single(0.40, 0.20, 0.45, -0.30, -1.0, 0.08))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"a": -1.0}, "a must be positive"),
        ({"t": -0.1}, "t must be non-negative"),
        ({"v_x": [0.1, np.nan]}, "v_x must contain only finite"),
        ({"theta": np.ones((2, 1))}, "theta must be a nonempty numeric scalar"),
    ],
)
def test_jax_fixed_cdm_eager_calls_validate_concrete_inputs(arguments, message):
    """Concrete misuse should fail before entering a traced computation."""
    inputs = {
        "rt": np.array([0.30, 0.50]),
        "theta": np.array([-0.4, 0.8]),
        "v_x": 0.45,
        "v_y": -0.30,
        "a": 1.20,
        "t": 0.08,
    }
    inputs.update(arguments)

    with pytest.raises(ValueError, match=message):
        fixed_cdm_logpdf(**inputs)


def test_jax_fixed_cdm_floor_does_not_mask_numerical_failure():
    """Post-validation flooring must leave a failed valid-support result visible."""
    arguments = {
        "rt": 1e-13,
        "theta": 0.2,
        "v_x": 0.45,
        "v_y": -0.30,
        "a": 1e-5,
        "t": 0.0,
    }

    strict = fixed_cdm_logpdf(**arguments)
    floored = fixed_cdm_logpdf(**arguments, log_density_floor=-66.1)

    assert np.isnan(strict[0])
    assert np.isnan(floored[0])


def test_base_import_is_jax_free_and_optional_import_keeps_precision_setting():
    """JAX should stay lazy and the optional module must not mutate global precision."""
    command = """
import sys
import jeam
assert "jax" not in sys.modules
import jax
jax.config.update("jax_enable_x64", False)
import jeam.likelihoods.jax_fixed
assert not jax.config.x64_enabled
"""
    subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
    )
