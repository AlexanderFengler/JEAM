import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from math import pi
    from pathlib import Path

    import marimo as mo
    import numpy as np

    _repository_root = str(Path(__file__).resolve().parents[1])
    if _repository_root not in sys.path:
        sys.path.insert(0, _repository_root)

    from jeam.Models.Circular import CircularDiffusionModel
    from jeam.Models.HyperSpherical import HyperSphericalDiffusionModel
    from jeam.Models.Spherical import SphericalDiffusionModel
    from tests.numerical_diagnostics import fixed_zero_drift_surface_density

    return (
        CircularDiffusionModel,
        HyperSphericalDiffusionModel,
        SphericalDiffusionModel,
        fixed_zero_drift_surface_density,
        mo,
        np,
        pi,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # JEAM failure-mode laboratory

    This executable notebook is the visual companion to JEAM's scientific hardening
    tests. The pytest suite owns every invariant; this laboratory explains those
    invariants and will accumulate one focused panel as each defect is corrected.

    **Current scope:** preserve the independently verified fixed-boundary core and make
    every audited failure visible. No red or amber row below is claimed fixed by this
    scaffold.
    """)
    return


@app.cell
def _(mo):
    failure_modes = [
        {
            "Area": "Fixed CDM / SDM / HSDM",
            "Audit result": "Independent Bessel-series baseline reproduced",
            "Invariant": "Pointwise density and radial mass",
            "Planned PR": "PR02",
            "Status": "✅ protected",
        },
        {
            "Area": "Density measure",
            "Audit result": "Angular-coordinate Jacobians are undocumented/absent",
            "Invariant": "Density integrates in the exposed coordinates",
            "Planned PR": "PR03",
            "Status": "🟠 pending",
        },
        {
            "Area": "Projected normalization",
            "Audit result": "PSDM and PHSDM omit dimensional constants",
            "Invariant": "Full-domain probability mass equals one",
            "Planned PR": "PR04",
            "Status": "🔴 failing",
        },
        {
            "Area": "PHSDM geometry",
            "Audit result": "Simulator and likelihood projections disagree",
            "Invariant": "Requested and reconstructed Cartesian drift agree",
            "Planned PR": "PR05",
            "Status": "🔴 failing",
        },
        {
            "Area": "PHSDM variability",
            "Audit result": "Combined s_v and s_t branch raises UnboundLocalError",
            "Invariant": "All four variability combinations are finite",
            "Planned PR": "PR06",
            "Status": "🔴 failing",
        },
        {
            "Area": "PHSDM sigma scaling",
            "Audit result": "Non-unit diffusion scaling is incomplete",
            "Invariant": "Equivalent rescaled parameterizations agree",
            "Planned PR": "PR07",
            "Status": "🔴 failing",
        },
        {
            "Area": "NDT leading edge",
            "Audit result": "Valid rt in (ndt, ndt + s_t] is floored",
            "Invariant": "Integrate the feasible uniform-NDT support",
            "Planned PR": "PR08",
            "Status": "🔴 failing",
        },
        {
            "Area": "Large-argument Bessel kernel",
            "Audit result": "Scaled branch uses the wrong argument and scale",
            "Invariant": "Agreement with arbitrary-precision Bessel evaluation",
            "Planned PR": "PR09",
            "Status": "🔴 failing",
        },
        {
            "Area": "API, RNG, and termination",
            "Audit result": "Validation, reproducibility, and loop bounds are incomplete",
            "Invariant": "Explicit domains, seeded simulation, bounded execution",
            "Planned PR": "PR10–PR13",
            "Status": "🟠 pending",
        },
    ]
    mo.ui.table(failure_modes, pagination=False, selection=None)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Protected fixed-boundary baseline

    Choose the accumulator dimension and decision time. JEAM's zero-drift density is
    compared pointwise with a test-only implementation of the paper's Bessel-zero
    series. The response angles are fixed at $\pi/2$, where both the natural-surface
    and ordinary-coordinate Jacobians equal one. This keeps the radial baseline valid
    while the public density measure is corrected separately.
    """)
    return


@app.cell
def _(mo):
    dimension_control = mo.ui.slider(
        start=2,
        stop=4,
        step=1,
        value=2,
        label="Accumulator dimension",
        show_value=True,
    )
    time_control = mo.ui.slider(
        start=0.05,
        stop=2.0,
        step=0.05,
        value=0.5,
        label="Decision time",
        show_value=True,
    )
    mo.hstack([dimension_control, time_control], justify="start")
    return dimension_control, time_control


@app.cell
def _(
    CircularDiffusionModel,
    HyperSphericalDiffusionModel,
    SphericalDiffusionModel,
    dimension_control,
    fixed_zero_drift_surface_density,
    mo,
    np,
    pi,
    time_control,
):
    _dimension = int(dimension_control.value)
    _decision_time = float(time_control.value)
    _model_types = {
        2: CircularDiffusionModel,
        3: SphericalDiffusionModel,
        4: HyperSphericalDiffusionModel,
    }
    _angles = np.full((1, _dimension - 1), pi / 2)
    if _dimension == 2:
        _angles = _angles[:, 0]
    _observed = float(
        np.exp(
            _model_types[_dimension]().joint_lpdf(
                rt=np.array([_decision_time]),
                theta=_angles,
                drift_vec=np.zeros(_dimension),
                ndt=0.0,
                threshold=1.0,
            )
        )[0]
    )
    _expected = float(fixed_zero_drift_surface_density(_dimension, [_decision_time])[0])
    _relative_error = abs(_observed - _expected) / _expected
    mo.ui.table(
        [
            {
                "Dimension": _dimension,
                "Decision time": _decision_time,
                "JEAM density": _observed,
                "Independent density": _expected,
                "Relative error": _relative_error,
                "Result": "PASS" if _relative_error < 2e-9 else "CHECK",
            }
        ],
        pagination=False,
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Density measure

    **Audit symptom:** SDM and HSDM currently return a density per unit natural surface
    area, while users supply ordinary angular coordinates. The required coordinate
    Jacobians are not part of the documented API.

    **Independent oracle:** integrate the returned density over the complete angular
    coordinate domain using explicit hyperspherical Jacobians. PR03 will turn this into
    a measure-declaration and coordinate-density panel.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Projected-model normalization

    **Audit symptom:** at zero drift, PSDM and PHSDM integrate to $(2\pi)^{3/2}$ and
    $4\pi^2$ under their current projected surface measures, not one.

    **Independent oracle:** full-domain quadrature with dimensional constants derived
    before calling JEAM. PR04 will plot cumulative mass before and after correction.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## PHSDM projection geometry

    **Audit symptom:** the asymmetric drift $[10, 1, 1]$ requests hidden radial
    component $1$, but the current angle reconstruction produces about $0.7106$; the
    simulator's first response angle also omits a coordinate.

    **Independent oracle:** Appendix A4's Cartesian projection. PR05 will visualize the
    requested vector, the current reconstruction, and the corrected shared mapping.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## PHSDM combined variability

    **Audit symptom:** the valid combination $s_v>0$ and $s_t>0$ references `p1`
    before assignment and mis-indexes trial-wise drift.

    **Independent oracle:** the four $(s_v, s_t)$ combinations must return one finite
    value per observation and converge to simpler branches as variability vanishes.
    PR06 will add that status panel.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## PHSDM diffusion scaling

    **Audit symptom:** the fixed PHSDM Girsanov time term and short-time branch omit
    required powers of $\sigma$.

    **Independent oracle:** analytically equivalent parameter rescalings. PR07 will
    plot the likelihood difference over several non-unit diffusion coefficients.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Non-decision-time leading edge

    **Audit symptom:** observations with $ndt < rt \le ndt+s_t$ retain feasible NDT
    support, but the current likelihood assigns its finite floor.

    **Independent oracle:** adaptive quadrature over only the feasible portion of the
    uniform support. PR08 will show the truncated integration interval and density.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Large-argument Bessel stability

    **Audit symptom:** above argument 700, the scaled-Bessel branch evaluates `ive` at
    a different argument and drops its compensating exponential. The audited stress
    case returned approximately $5.17\times10^{-4}$ instead of
    $-1.38\times10^{-115}$.

    **Independent oracle:** arbitrary-precision `mpmath` evaluation and SciPy's scaled
    Bessel function. PR09 will compare both sides of the branch threshold.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## API domains, RNG, and termination

    **Audit symptom:** advertised list inputs can fail before validation; impossible
    observations and numerical failures share a finite floor; simulation has neither
    injected RNG state nor a non-crossing-boundary limit.

    **Independent oracle:** explicit input/output contracts, same-seed equality,
    different-seed inequality, and a targeted termination exception. PR10–PR13 will
    separate these concerns into reviewable panels.
    """)
    return


if __name__ == "__main__":
    app.run()
