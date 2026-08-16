"""Value and gradient contract for the fixed-CDM JAX likelihood."""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from jeam.Models.Circular import CircularDiffusionModel

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
        rt=0.108728,
        theta=0.35,
        parameters=(-0.20, 0.65, 1.20, 0.08),
        scaled_decision_time=0.01995,
        expected_logpdf=-19.410805657275215,
        expected_gradient=(
            1.132992855934933,
            0.3928041690137711,
            -40.0690215002112,
            -802.5316709872933,
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
