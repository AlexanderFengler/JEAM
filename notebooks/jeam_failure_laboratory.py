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
    from jeam.Models.HyperSpherical import (
        HyperSphericalDiffusionModel,
        ProjectedHyperSphericalDiffusionModel,
    )
    from jeam.Models.Spherical import (
        ProjectedSphericalDiffusionModel,
        SphericalDiffusionModel,
    )
    from tests.numerical_diagnostics import (
        fixed_zero_drift_surface_density,
        trapezoid_mass,
        unit_sphere_surface_jacobian,
    )

    return (
        CircularDiffusionModel,
        HyperSphericalDiffusionModel,
        ProjectedHyperSphericalDiffusionModel,
        ProjectedSphericalDiffusionModel,
        SphericalDiffusionModel,
        fixed_zero_drift_surface_density,
        mo,
        np,
        pi,
        trapezoid_mass,
        unit_sphere_surface_jacobian,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # JEAM failure-mode laboratory

    This executable notebook is the visual companion to JEAM's scientific hardening
    tests. The pytest suite owns every invariant; this laboratory explains those
    invariants and will accumulate one focused panel as each defect is corrected.

    **Current scope:** preserve the independently verified fixed-boundary core, explain
    the corrected density contracts and reproducible CDM simulator, and keep every
    remaining audited failure visible.
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
            "Audit result": "Coordinate Jacobians included and documented",
            "Invariant": "Density integrates in the exposed coordinates",
            "Planned PR": "PR03",
            "Status": "✅ protected",
        },
        {
            "Area": "Projected normalization",
            "Audit result": "Dimensional constants and Jacobians restored",
            "Invariant": "Full-domain probability mass equals one",
            "Planned PR": "PR04",
            "Status": "✅ protected",
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
            "Area": "API and numerical errors",
            "Audit result": "Validation and error policy remain incomplete",
            "Invariant": "Explicit domains and distinct support/numerical failures",
            "Planned PR": "PR10–PR11",
            "Status": "🟠 pending",
        },
        {
            "Area": "CDM simulation RNG",
            "Audit result": "Generator state is injected through each CDM batch",
            "Invariant": "Same seed reproduces without global RNG mutation",
            "Planned PR": "PR12a",
            "Status": "✅ protected",
        },
        {
            "Area": "Other simulator RNG",
            "Audit result": "Other model families still use process-global state",
            "Invariant": "Apply the accepted CDM contract after scientific fixes",
            "Planned PR": "PR12b",
            "Status": "🟠 pending",
        },
        {
            "Area": "Simulation termination",
            "Audit result": "Custom boundaries have no non-crossing limit",
            "Invariant": "Fail explicitly at a configured maximum time or step",
            "Planned PR": "PR13",
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

    **Pre-fix symptom:** SDM and HSDM returned a density per unit natural surface area,
    while users supplied ordinary angular coordinates. Integrating those values directly
    over the rectangular angle domains over-counted the angular mass.

    **Corrected contract:** `joint_lpdf` is a density with respect to ordinary Lebesgue
    measure in the exposed columns. SDM therefore includes
    $\sin(\theta_1)$, and HSDM includes
    $\sin^2(\theta_1)\sin(\theta_2)$. The panel compares JEAM pointwise with the
    independent surface-density oracle times that Jacobian, then integrates JEAM over
    the complete angular domain. The first table reconstructs the historical formula
    without the Jacobian and places its mass ratio beside the corrected result for both
    models.
    """)
    return


@app.cell
def _(
    HyperSphericalDiffusionModel,
    SphericalDiffusionModel,
    fixed_zero_drift_surface_density,
    mo,
    np,
    pi,
    trapezoid_mass,
):
    _comparison_time = 0.5
    _comparison_cases = [
        (
            "SDM",
            3,
            SphericalDiffusionModel,
            (np.linspace(0.0, pi, 101), np.linspace(-pi, pi, 2)),
            4 * pi,
        ),
        (
            "HSDM",
            4,
            HyperSphericalDiffusionModel,
            (
                np.linspace(0.0, pi, 101),
                np.linspace(0.0, pi, 101),
                np.linspace(-pi, pi, 2),
            ),
            2 * pi**2,
        ),
    ]
    _comparison_rows = []
    for _model_name, _dimension, _model_type, _axes, _surface_area in _comparison_cases:
        _surface_density = float(
            fixed_zero_drift_surface_density(_dimension, [_comparison_time])[0]
        )
        _angular_grid = np.meshgrid(*_axes, indexing="ij")
        _angles = np.column_stack(
            [_coordinate.ravel() for _coordinate in _angular_grid]
        )
        _corrected_density = np.exp(
            _model_type().joint_lpdf(
                rt=np.full(_angles.shape[0], _comparison_time),
                theta=_angles,
                drift_vec=np.zeros(_dimension),
                ndt=0.0,
                threshold=1.0,
            )
        ).reshape(_angular_grid[0].shape)
        _historical_density = np.full_like(_corrected_density, _surface_density)
        _expected_mass = _surface_area * _surface_density
        _historical_ratio = trapezoid_mass(_historical_density, _axes) / _expected_mass
        _corrected_ratio = trapezoid_mass(_corrected_density, _axes) / _expected_mass
        _comparison_rows.append(
            {
                "Model": _model_name,
                "Historical pre-fix mass ratio": _historical_ratio,
                "Corrected JEAM mass ratio": _corrected_ratio,
                "Expected ratio": 1.0,
                "Resolution": (
                    "RESOLVED"
                    if abs(_corrected_ratio - 1.0) < 2e-4
                    and abs(_historical_ratio - 1.0) > 0.1
                    else "CHECK"
                ),
            }
        )

    mo.vstack(
        [
            mo.md(
                "**Resolved defect at decision time 0.5.** The pre-fix column "
                "reconstructs the audited implementation by integrating the independent "
                "surface density directly over the exposed rectangular angle domain."
            ),
            mo.ui.table(_comparison_rows, pagination=False, selection=None),
        ]
    )
    return


@app.cell
def _(mo, pi):
    density_dimension_control = mo.ui.slider(
        start=3,
        stop=4,
        step=1,
        value=3,
        label="Accumulator dimension",
        show_value=True,
    )
    primary_polar_control = mo.ui.slider(
        start=0.05,
        stop=pi - 0.05,
        step=0.05,
        value=1.05,
        label="Primary polar angle",
        show_value=True,
    )
    secondary_polar_control = mo.ui.slider(
        start=0.05,
        stop=pi - 0.05,
        step=0.05,
        value=0.8,
        label="Second polar angle (HSDM only)",
        show_value=True,
    )
    mo.vstack(
        [
            density_dimension_control,
            primary_polar_control,
            secondary_polar_control,
        ],
        align="start",
    )
    return (
        density_dimension_control,
        primary_polar_control,
        secondary_polar_control,
    )


@app.cell
def _(
    HyperSphericalDiffusionModel,
    SphericalDiffusionModel,
    density_dimension_control,
    fixed_zero_drift_surface_density,
    mo,
    np,
    pi,
    primary_polar_control,
    secondary_polar_control,
    time_control,
    trapezoid_mass,
    unit_sphere_surface_jacobian,
):
    _dimension = int(density_dimension_control.value)
    _decision_time = float(time_control.value)
    _model_types = {
        3: SphericalDiffusionModel,
        4: HyperSphericalDiffusionModel,
    }
    _surface_areas = {3: 4 * pi, 4: 2 * pi**2}
    if _dimension == 3:
        _selected_angles = np.array([[primary_polar_control.value, 0.0]])
        _axes = (
            np.linspace(0.0, pi, 101),
            np.linspace(-pi, pi, 2),
        )
    else:
        _selected_angles = np.array(
            [
                [
                    primary_polar_control.value,
                    secondary_polar_control.value,
                    0.0,
                ]
            ]
        )
        _axes = (
            np.linspace(0.0, pi, 81),
            np.linspace(0.0, pi, 81),
            np.linspace(-pi, pi, 2),
        )

    _surface_density = float(
        fixed_zero_drift_surface_density(_dimension, [_decision_time])[0]
    )
    _coordinate_jacobian = float(unit_sphere_surface_jacobian(*_selected_angles.T)[0])
    _observed_density = float(
        np.exp(
            _model_types[_dimension]().joint_lpdf(
                rt=np.array([_decision_time]),
                theta=_selected_angles,
                drift_vec=np.zeros(_dimension),
                ndt=0.0,
                threshold=1.0,
            )
        )[0]
    )
    _expected_density = _surface_density * _coordinate_jacobian

    _angular_grid = np.meshgrid(*_axes, indexing="ij")
    _all_angles = np.column_stack(
        [_coordinate.ravel() for _coordinate in _angular_grid]
    )
    _grid_density = np.exp(
        _model_types[_dimension]().joint_lpdf(
            rt=np.full(_all_angles.shape[0], _decision_time),
            theta=_all_angles,
            drift_vec=np.zeros(_dimension),
            ndt=0.0,
            threshold=1.0,
        )
    ).reshape(_angular_grid[0].shape)
    _observed_mass = trapezoid_mass(_grid_density, _axes)
    _expected_mass = _surface_areas[_dimension] * _surface_density

    mo.ui.table(
        [
            {
                "Model": "SDM" if _dimension == 3 else "HSDM",
                "Coordinate Jacobian": _coordinate_jacobian,
                "JEAM coordinate density": _observed_density,
                "Oracle × Jacobian": _expected_density,
                "Pointwise relative error": abs(_observed_density - _expected_density)
                / _expected_density,
                "Integrated angular mass": _observed_mass,
                "Expected angular mass": _expected_mass,
                "Mass ratio": _observed_mass / _expected_mass,
            }
        ],
        pagination=False,
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Projected-model normalization

    **Pre-fix symptom:** at zero drift, PSDM and PHSDM integrated to
    $(2\pi)^{3/2}$ and $4\pi^2$ under their projected surface measures, not one.
    They also omitted the remaining response-coordinate Jacobians after marginalizing
    the hidden azimuth.

    **Corrected contract:** the projected likelihoods return densities in their exposed
    response coordinates. The table integrates over response time and the full projected
    response domain. Its historical column reconstructs the audited implementation by
    reversing the missing dimensional constant at unit-Jacobian response points; its
    corrected column calls JEAM directly. The independent formula remains enforced in
    pytest.
    """)
    return


@app.cell
def _(
    ProjectedHyperSphericalDiffusionModel,
    ProjectedSphericalDiffusionModel,
    mo,
    np,
    pi,
    trapezoid_mass,
):
    _projected_times = np.linspace(1e-6, 12.0, 30_000)
    _projected_cases = [
        (
            "PSDM",
            3,
            ProjectedSphericalDiffusionModel,
            np.full(_projected_times.size, pi / 2),
            2.0,
            (2 * pi) ** 1.5,
        ),
        (
            "PHSDM",
            4,
            ProjectedHyperSphericalDiffusionModel,
            np.full((_projected_times.size, 2), pi / 2),
            pi,
            4 * pi**2,
        ),
    ]
    _projected_rows = []
    for (
        _model_name,
        _dimension,
        _model_type,
        _unit_jacobian_angles,
        _response_jacobian_integral,
        _audited_historical_mass,
    ) in _projected_cases:
        _corrected_density = np.exp(
            _model_type().joint_lpdf(
                rt=_projected_times,
                theta=_unit_jacobian_angles,
                drift_vec=np.zeros(_dimension - 1),
                ndt=0.0,
                threshold=1.0,
            )
        )
        _historical_density = _audited_historical_mass * _corrected_density
        _historical_mass = _response_jacobian_integral * trapezoid_mass(
            _historical_density, (_projected_times,)
        )
        _corrected_mass = _response_jacobian_integral * trapezoid_mass(
            _corrected_density, (_projected_times,)
        )
        _projected_rows.append(
            {
                "Model": _model_name,
                "Historical pre-fix mass": _historical_mass,
                "Corrected JEAM mass": _corrected_mass,
                "Expected mass": 1.0,
                "Resolution": (
                    "RESOLVED"
                    if abs(_corrected_mass - 1.0) < 2e-4
                    and abs(_historical_mass - _audited_historical_mass) < 2e-3
                    else "CHECK"
                ),
            }
        )

    mo.ui.table(_projected_rows, pagination=False, selection=None)
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
    ## Reproducible CDM simulation

    **Audited defect:** the pre-fix CDM simulator exposed no `random_state` argument and
    drew inside its compiled helper from process-global `np.random` state. Calling
    `simulate(..., random_state=1947)` therefore raised `TypeError`; a caller could not
    reproduce a complete batch through the public API.

    **Corrected contract:** an integer seed or `numpy.random.Generator` is normalized at
    the batch boundary and the same Generator stream is advanced across every trial.
    The checks below compare complete outputs exactly and independently verify that the
    legacy global NumPy state is unchanged.
    """)
    return


@app.cell
def _(mo):
    rng_seed_control = mo.ui.slider(
        start=1,
        stop=10_000,
        step=1,
        value=1947,
        label="CDM seed",
        show_value=True,
    )
    rng_seed_control
    return (rng_seed_control,)


@app.cell
def _(CircularDiffusionModel, mo, np, rng_seed_control):
    _seed = int(rng_seed_control.value)
    _arguments = {
        "drift_vec": np.array([0.7, -0.35]),
        "ndt": 0.2,
        "threshold": 0.45,
        "s_v": 0.1,
        "s_t": 0.05,
        "sigma": 0.8,
        "dt": 0.005,
        "n_sample": 6,
    }

    _model = CircularDiffusionModel()
    _first = _model.simulate(**_arguments, random_state=_seed)
    _matching = _model.simulate(**_arguments, random_state=_seed)
    _different = _model.simulate(**_arguments, random_state=_seed + 1)

    _first_generator = np.random.default_rng(_seed)
    _second_generator = np.random.default_rng(_seed)
    _generator_first = _model.simulate(**_arguments, random_state=_first_generator)
    _generator_matching = _model.simulate(**_arguments, random_state=_second_generator)
    _generator_advanced = _model.simulate(**_arguments, random_state=_first_generator)

    _original_legacy_state = np.random.get_state()
    try:
        np.random.seed(831)
        _legacy_before = np.random.get_state()
        _model.simulate(**_arguments, random_state=_seed)
        _legacy_after = np.random.get_state()
        _legacy_unchanged = (
            _legacy_before[0] == _legacy_after[0]
            and np.array_equal(_legacy_before[1], _legacy_after[1])
            and _legacy_before[2:] == _legacy_after[2:]
        )
    finally:
        np.random.set_state(_original_legacy_state)

    _same_seed = _first.equals(_matching)
    _different_seed = not _first.equals(_different)
    _fresh_generators_match = _generator_first.equals(_generator_matching)
    _reused_generator_advances = not _generator_first.equals(_generator_advanced)
    _one_stream_per_batch = len(_first.drop_duplicates()) > 1

    _rng_checks = [
        {
            "Invariant": "Public seed argument",
            "Audited before PR12a": "TypeError: random_state was not accepted",
            "Current result": "PASS" if _same_seed else "FAIL",
        },
        {
            "Invariant": "Different integer seeds change the batch",
            "Audited before PR12a": "No public seed contract",
            "Current result": "PASS" if _different_seed else "FAIL",
        },
        {
            "Invariant": "Fresh equal Generators reproduce",
            "Audited before PR12a": "Generator injection unavailable",
            "Current result": "PASS" if _fresh_generators_match else "FAIL",
        },
        {
            "Invariant": "A reused Generator advances",
            "Audited before PR12a": "Generator injection unavailable",
            "Current result": "PASS" if _reused_generator_advances else "FAIL",
        },
        {
            "Invariant": "One stream advances across batch trials",
            "Audited before PR12a": "Process-global draws inside each helper",
            "Current result": "PASS" if _one_stream_per_batch else "FAIL",
        },
        {
            "Invariant": "Legacy np.random state is unchanged",
            "Audited before PR12a": "Process-global RNG was the only source",
            "Current result": "PASS" if _legacy_unchanged else "FAIL",
        },
    ]
    _all_resolved = all(row["Current result"] == "PASS" for row in _rng_checks)

    mo.vstack(
        [
            mo.callout(
                "RESOLVED: all CDM RNG invariants pass."
                if _all_resolved
                else "UNRESOLVED: at least one CDM RNG invariant failed.",
                kind="success" if _all_resolved else "danger",
            ),
            mo.ui.table(_rng_checks, pagination=False, selection=None),
            mo.md("**Same-seed batch preview**"),
            mo.ui.table(
                _first.assign(
                    matching_rt=_matching["rt"],
                    matching_response=_matching["response"],
                ),
                pagination=False,
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Remaining API and simulation safety work

    List coercion and parameter validation remain assigned to PR10; support versus
    numerical-error behavior remains assigned to PR11. PR12b will extend the accepted
    RNG contract to other model families after their scientific fixes, and PR13 will add
    a targeted exception for custom boundaries that never cross.
    """)
    return


if __name__ == "__main__":
    app.run()
