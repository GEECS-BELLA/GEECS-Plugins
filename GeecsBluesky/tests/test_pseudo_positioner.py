"""CaPseudoPositioner: the pseudo scan variable as a pseudo positioner over its components.

Mock-backend tests (no gateway): the components are ``CaMotor``s whose
mock readbacks follow their ``:SP`` puts (GEECS's native convergence
stand-in), the pseudo is built from a catalog document through
:func:`build_pseudo` exactly as the namespace will build it, and scans are
driven through a real RunEngine so stage → move → unstage happens the way
the stock plans do it.  Pins ``09_pseudo_transform.md`` §3: zeroing at
stage, the derived readback, the disagreement check's fail/warn split, the
restore at unstage (end of scan and abort), and the build-time refusals.
"""

from __future__ import annotations

import asyncio
import logging
import math

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
import bluesky.plans as bp  # noqa: E402
import bluesky.preprocessors as bpp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from bluesky.utils import FailedStatus  # noqa: E402
from geecs_schemas.scan_variables import ScanVariables  # noqa: E402
from ophyd_async.core import callback_on_mock_put, set_mock_value  # noqa: E402

from geecs_bluesky.devices.ca import CaMotor, CaPseudoPositioner, CaSettable  # noqa: E402
from geecs_bluesky.devices.ca.pseudo import (  # noqa: E402
    DEFAULT_AGREEMENT_TOLERANCE,
    build_pseudo,
    inverse_symbols,
)
from geecs_bluesky.preprocessors import scalar_headers  # noqa: E402
from geecs_bluesky.exceptions import (  # noqa: E402
    GeecsConfigurationError,
    PseudoComponentsDisagreeError,
)
from tests.ca_mock_helpers import DocCollector, connect_mock  # noqa: E402

R56_FORWARD = "sqrt(100 ** 2 * composite_var / 560968.636)"
R56_INVERSE = "560968.636 * U_ChicaneInner**2 / 100**2"

CATALOG = ScanVariables.model_validate(
    {
        "schema_version": 1,
        "variables": {
            "ALine_e_beam_angle_offset_x": {
                "kind": "pseudo",
                "mode": "relative",
                "targets": [
                    {"target": "U_S3H:Current", "forward": "composite_var * 1"},
                    {"target": "U_S4H:Current", "forward": "composite_var * -2"},
                ],
            },
            "R56_at_100MeV": {
                "kind": "pseudo",
                "mode": "absolute",
                "targets": [
                    {"target": "U_ChicaneInner:Current", "forward": R56_FORWARD},
                    {"target": "U_ChicaneOuter:Current", "forward": "-" + R56_FORWARD},
                ],
                "inverse": R56_INVERSE,
            },
            "JetZ_with_probe": {
                "kind": "pseudo",
                "mode": "absolute",
                "targets": [
                    {
                        "target": "U_ESP_JetXYZ:Position.Axis 3",
                        "forward": "composite_var",
                    },
                    {
                        "target": "U_ProbeCamStage:Position",
                        "forward": "8.5 + (composite_var-10)*2.5",
                    },
                ],
            },
        },
    }
).variables

TARGETS = [
    "U_S3H:Current",
    "U_S4H:Current",
    "U_ChicaneInner:Current",
    "U_ChicaneOuter:Current",
    "U_ESP_JetXYZ:Position.Axis 3",
    "U_ProbeCamStage:Position",
]


class Bench:
    """Mock components + a journal of every ``:SP`` put, on a RunEngine loop."""

    def __init__(self, run_engine: RunEngine) -> None:
        self.RE = run_engine
        self.journal: list[tuple[str, float]] = []
        self.components: dict[str, CaMotor] = {}
        for target in TARGETS:
            device, variable = target.split(":")
            motor = CaMotor(device, variable, tolerance=0.005, name=device.lower())
            connect_mock(run_engine, motor)

            def _follow(value, *, motor=motor, target=target, **kwargs):
                self.journal.append((target, value))
                set_mock_value(motor.position, value)

            callback_on_mock_put(motor._setpoint, _follow)
            self.components[target] = motor

    def build(self, name: str, **kwargs) -> CaPseudoPositioner:
        pseudo = build_pseudo(
            name, CATALOG[name], self.components.__getitem__, **kwargs
        )
        connect_mock(self.RE, pseudo)
        return pseudo

    def place(self, **positions: float) -> None:
        """Move mock readbacks by hand (a LabVIEW move, an operator)."""
        for device, value in positions.items():
            target = next(t for t in TARGETS if t.split(":")[0] == device)
            set_mock_value(self.components[target].position, value)

    def dial(self, device: str) -> float:
        target = next(t for t in TARGETS if t.split(":")[0] == device)
        return self.run(self.components[target].position.get_value())

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.RE._loop).result(
            timeout=10.0
        )


@pytest.fixture
def bench() -> Bench:
    return Bench(RunEngine())


# ----------------------------------------------------------- relative (bump)


def test_relative_zeroes_at_stage_reads_zero_and_moves_about_the_baselines(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)  # today's alignment, not in the 1:-2 ratio

    def plan():
        yield from bps.stage(bump, wait=True)
        location = yield from bps.locate(bump)
        assert location == {"setpoint": 0.0, "readback": 0.0}
        yield from bps.mv(bump, 0.1)
        reading = yield from bps.rd(bump)
        assert reading == pytest.approx(0.1)
        yield from bps.unstage(bump, wait=True)

    bench.RE(plan())

    moves = [(t, v) for t, v in bench.journal]
    assert moves[:2] == [
        ("U_S3H:Current", pytest.approx(0.45)),
        ("U_S4H:Current", pytest.approx(-0.299)),
    ]
    # unstage restored the captured baselines exactly — through the components' set()
    assert moves[2:] == [
        ("U_S3H:Current", pytest.approx(0.35)),
        ("U_S4H:Current", pytest.approx(-0.099)),
    ]
    assert bench.dial("U_S3H") == pytest.approx(0.35)
    assert bench.dial("U_S4H") == pytest.approx(-0.099)
    assert bump.last_commanded == {
        "U_S3H:Current": pytest.approx(0.35),
        "U_S4H:Current": pytest.approx(-0.099),
    }


def test_relative_offsets_live_on_the_components(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)
    s3h = bench.components["U_S3H:Current"]
    assert bench.run(s3h.offset.get_value()) == 0.0
    bench.RE(bps.stage(bump, wait=True))
    assert bench.run(s3h.offset.get_value()) == pytest.approx(
        -0.35
    )  # user = dial + offset = 0
    assert bench.run(
        bench.components["U_S4H:Current"].offset.get_value()
    ) == pytest.approx(0.099)
    bench.RE(bps.unstage(bump, wait=True))


def test_set_zero_restores_and_a_scan_restores_at_its_end(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)
    collector = DocCollector()

    bench.RE(scalar_headers(bp.scan([], bump, -0.1, 0.1, 3)), collector)

    values = [
        e["data"]["aline_e_beam_angle_offset_x-readback"]
        for e in collector.primary_events()
    ]
    assert values == pytest.approx([-0.1, 0.0, 0.1])
    assert bench.dial("U_S3H") == pytest.approx(0.35)
    assert bench.dial("U_S4H") == pytest.approx(-0.099)
    assert collector.docs["start"][0]["geecs_scalar_headers"] == {
        "aline_e_beam_angle_offset_x-readback": "ALine_e_beam_angle_offset_x"
    }


def test_rel_scan_over_a_relative_pseudo_is_the_same_scan(bench):
    """The #855 case: rel_scan locates 0 at the first set (after stage), so its
    put-back at the end is set(0) — the moves and the restore of a plain scan."""
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)

    bench.RE(bp.rel_scan([], bump, -0.2, 0.2, 3))

    s3h = [v for t, v in bench.journal if t == "U_S3H:Current"]
    assert s3h == pytest.approx(
        [0.15, 0.35, 0.55, 0.35, 0.35]
    )  # 3 points, restore, put-back
    assert bench.dial("U_S4H") == pytest.approx(-0.099)


def test_aborted_relative_scan_restores_the_baselines(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)

    def plan():
        yield from bps.mv(bump, 0.2)
        raise RuntimeError("operator stop")  # the RE aborts; unstage is a finalize

    with pytest.raises(RuntimeError, match="operator stop"):
        # the stock plans' own bracket: stage_all … unstage_all as a finalize
        bench.RE(bpp.stage_wrapper(plan(), [bump]))
    assert bench.journal[-2:] == [
        ("U_S3H:Current", pytest.approx(0.35)),
        ("U_S4H:Current", pytest.approx(-0.099)),
    ]

    assert bench.dial("U_S3H") == pytest.approx(0.35)
    assert bench.dial("U_S4H") == pytest.approx(-0.099)


def test_relative_component_moved_under_the_scan_fails_before_moving_anything(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)

    def plan():
        yield from bps.stage(bump, wait=True)
        yield from bps.mv(bump, 0.1)
        bench.place(U_S4H=-0.5)  # a hand move mid-scan
        bench.journal.clear()
        yield from bps.mv(bump, 0.2)

    with pytest.raises(FailedStatus) as info:
        bench.RE(plan())
    cause = info.value.__cause__
    assert isinstance(cause, PseudoComponentsDisagreeError)
    assert "U_S4H:Current reads -0.5" in str(cause)
    # The step itself never moved anything; the only puts that may follow
    # are the RE's abort-time unstage restoring the baselines.
    moves = {(t, round(v, 6)) for t, v in bench.journal}
    assert ("U_S3H:Current", 0.55) not in moves and (
        "U_S4H:Current",
        -0.499,
    ) not in moves
    assert moves <= {("U_S3H:Current", 0.35), ("U_S4H:Current", -0.099)}


def test_relative_disagreement_within_tolerance_is_not_a_disagreement(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x", tolerance=lambda t: 0.02)
    bench.place(U_S3H=0.35, U_S4H=-0.099)

    def plan():
        yield from bps.stage(bump, wait=True)
        yield from bps.mv(bump, 0.1)
        bench.place(U_S4H=-0.299 + 0.015)  # readback scatter inside the tolerance
        yield from bps.mv(bump, 0.2)
        yield from bps.unstage(bump, wait=True)

    bench.RE(plan())


def test_unstaged_manual_move_captures_todays_positions_and_set_zero_is_no_move(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)

    bench.RE(bps.mv(bump, 0.05))
    assert bench.dial("U_S3H") == pytest.approx(0.40)
    assert bench.dial("U_S4H") == pytest.approx(-0.199)
    bench.RE(bps.mv(bump, 0.0))
    assert bench.dial("U_S3H") == pytest.approx(0.35)
    assert bench.dial("U_S4H") == pytest.approx(-0.099)


def test_a_later_stage_takes_the_current_positions_as_the_new_alignment(bench):
    bump = bench.build("ALine_e_beam_angle_offset_x")
    bench.place(U_S3H=0.35, U_S4H=-0.099)
    bench.RE(bps.mv(bump, 0.05))  # a manual bump left in place: the operator aligned
    bench.RE(bp.scan([], bump, -0.1, 0.1, 3))
    assert bench.dial("U_S3H") == pytest.approx(0.40)
    assert bench.dial("U_S4H") == pytest.approx(-0.199)


# --------------------------------------------------- plain pseudo positioner


def test_plain_pseudo_locates_through_the_inverse_and_snaps_at_the_first_step(
    bench, caplog
):
    r56 = bench.build("R56_at_100MeV")
    bench.place(
        U_ChicaneInner=1.0, U_ChicaneOuter=-0.9
    )  # outer off the formula by hand

    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.ca.pseudo"):
        location = bench.run(r56.locate())
    assert location["readback"] == pytest.approx(56.0968636)
    assert "U_ChicaneOuter:Current reads -0.9, formula says -1" in caplog.text

    bench.RE(bps.mv(r56, 2.0))
    expected = math.sqrt(100**2 * 2.0 / 560968.636)
    assert bench.dial("U_ChicaneInner") == pytest.approx(expected)
    assert bench.dial("U_ChicaneOuter") == pytest.approx(-expected)
    assert bench.run(r56.readback.get_value()) == pytest.approx(2.0)


def test_plain_pseudo_component_moved_after_the_first_step_fails(bench):
    r56 = bench.build("R56_at_100MeV")
    bench.place(U_ChicaneInner=1.0, U_ChicaneOuter=-1.0)

    def plan():
        yield from bps.stage(r56, wait=True)
        yield from bps.mv(r56, 2.0)
        bench.place(U_ChicaneOuter=-0.5)
        yield from bps.mv(r56, 3.0)

    with pytest.raises(FailedStatus) as info:
        bench.RE(plan())
    assert isinstance(info.value.__cause__, PseudoComponentsDisagreeError)


def test_plain_pseudo_ends_where_it_is_and_rel_scan_puts_it_back(bench):
    r56 = bench.build("R56_at_100MeV")
    bench.place(U_ChicaneInner=1.0, U_ChicaneOuter=-1.0)  # R56 = 56.0968636

    bench.RE(bp.scan([], r56, 2.0, 4.0, 3))
    assert bench.run(r56.readback.get_value()) == pytest.approx(4.0)

    bench.RE(bp.rel_scan([], r56, -1.0, 1.0, 3))
    assert bench.run(r56.readback.get_value()) == pytest.approx(4.0)


def test_affine_plain_pseudo_inverts_through_its_identity_component(bench):
    jet = bench.build("JetZ_with_probe")
    bench.place(U_ESP_JetXYZ=12.0, U_ProbeCamStage=13.5)  # on the formula: 8.5 + 2*2.5
    assert bench.run(jet.locate())["readback"] == pytest.approx(12.0)
    bench.RE(bps.mv(jet, 10.0))
    assert bench.dial("U_ProbeCamStage") == pytest.approx(8.5)


def test_out_of_domain_value_is_refused_naming_the_component(bench):
    r56 = bench.build("R56_at_100MeV")
    bench.place(U_ChicaneInner=1.0, U_ChicaneOuter=-1.0)
    with pytest.raises(FailedStatus) as info:
        bench.RE(bps.mv(r56, -1.0))
    assert "R56_at_100MeV: no setting for -1.0" in str(info.value.__cause__)
    assert "math domain error" in str(info.value.__cause__)
    with pytest.raises(FailedStatus):
        bench.RE(bps.mv(r56, math.nan))


# ---------------------------------------------------------------- building


def _entry(**overrides):
    base = {
        "kind": "pseudo",
        "mode": "relative",
        "targets": [
            {"target": "U_S3H:Current", "forward": "x"},
            {"target": "U_S4H:Current", "forward": "x * -2"},
        ],
    }
    base.update(overrides)
    return ScanVariables.model_validate(
        {"schema_version": 1, "variables": {"p": base}}
    ).variables["p"]


def test_build_refuses_a_relative_forward_that_is_not_zero_at_zero(bench):
    spec = _entry(targets=[{"target": "U_S3H:Current", "forward": "x + 1"}])
    with pytest.raises(GeecsConfigurationError, match="not 0 at 0"):
        build_pseudo("p", spec, bench.components.__getitem__)


def test_build_refuses_a_non_affine_forward_without_an_inverse(bench):
    spec = _entry(
        mode="absolute", targets=[{"target": "U_S3H:Current", "forward": "x ** 2"}]
    )
    with pytest.raises(GeecsConfigurationError, match="not affine.*no 'inverse'"):
        build_pseudo("p", spec, bench.components.__getitem__)


def test_build_refuses_an_inverse_that_does_not_undo_the_forward(bench):
    spec = _entry(mode="absolute", inverse="U_S3H * 3")
    with pytest.raises(GeecsConfigurationError, match="does not undo"):
        build_pseudo("p", spec, bench.components.__getitem__)


def test_build_refuses_a_duplicate_target_and_a_non_settable(bench):
    spec = _entry(
        targets=[
            {"target": "U_S3H:Current", "forward": "x"},
            {"target": "U_S3H:Current", "forward": "x"},
        ]
    )
    with pytest.raises(GeecsConfigurationError, match="listed twice"):
        build_pseudo("p", spec, bench.components.__getitem__)
    with pytest.raises(GeecsConfigurationError, match="not a settable numeric"):
        build_pseudo("p", _entry(), lambda target: object())


def test_build_tolerances_follow_the_db_and_fall_back(bench):
    pseudo = build_pseudo(
        "p",
        _entry(),
        bench.components.__getitem__,
        tolerance={"U_S3H:Current": 0.05, "U_S4H:Current": 0.0}.__getitem__,
    )
    assert pseudo._tolerances == [0.05, DEFAULT_AGREEMENT_TOLERANCE]


def test_inverse_symbols_offer_the_device_name_when_unique_and_the_full_target():
    symbols = inverse_symbols(
        ["U_ESP302_02:Position.Axis 3", "U_ESP302_02:Position.Axis 1", "U_S1H:Current"]
    )
    assert symbols == {
        "U_ESP302_02_Position_Axis_3": "u_esp302_02_position_axis_3",
        "U_ESP302_02_Position_Axis_1": "u_esp302_02_position_axis_1",
        "U_S1H_Current": "u_s1h_current",
        "U_S1H": "u_s1h_current",
    }


def test_build_names_the_device_after_the_catalog_key(bench):
    pseudo = bench.build("ALine_e_beam_angle_offset_x")
    assert pseudo.name == "aline_e_beam_angle_offset_x"
    assert pseudo.relative is True
    assert pseudo._column_headers == {
        "aline_e_beam_angle_offset_x-readback": "ALine_e_beam_angle_offset_x"
    }
    assert bench.build("R56_at_100MeV").relative is False


# --------------------------------------------------------- component offset


async def test_settable_offset_defaults_to_zero_and_set_current_position_moves_nothing():
    settable = CaSettable("U_S3H", "Current", name="s3h")
    await settable.connect(mock=True)
    set_mock_value(settable.readback, 0.35)
    puts: list[float] = []
    callback_on_mock_put(settable._setpoint, lambda v, **kw: puts.append(v))

    assert await settable.offset.get_value() == 0.0
    await settable.set_current_position(0.0)
    assert await settable.offset.get_value() == pytest.approx(-0.35)
    await settable.set_current_position(1.0)
    assert await settable.offset.get_value() == pytest.approx(0.65)
    assert puts == []
    # the offset is not a readable: the event column stays the dial readback
    assert set(await settable.read()) == {"s3h-readback"}
    assert (await settable.locate())["readback"] == pytest.approx(0.35)


def test_affine_inverse_reads_the_identity_component(bench, caplog):
    """The identity target *is* the value: a disagreement is reported at that value."""
    jet = bench.build("JetZ_with_probe")
    bench.place(U_ESP_JetXYZ=12.0, U_ProbeCamStage=20.0)  # probe stage off the formula
    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.devices.ca.pseudo"):
        location = bench.run(jet.locate())
    assert location["readback"] == pytest.approx(12.0)
    assert "U_ProbeCamStage:Position reads 20, formula says 13.5 at 12" in caplog.text
