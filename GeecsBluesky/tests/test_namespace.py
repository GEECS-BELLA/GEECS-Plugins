"""GeecsNamespace — the experiment's devices as nouns, composed from the ca layer (#807 phase 1).

Built from an explicit roster (no DB); the DB path is exercised with a fake
``GeecsDb`` so the loud-failure contract is pinned without a database.
Pins the hardware-derived rules of
``Planning/native_bluesky/01_device_namespace.md``: served-set children,
DB-derived types, protocol-name collisions, the triggerable shortcut.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "aioca"
)  # CA backend needs the `ca` extra (CI's pure-unit job lacks it)
from ophyd_async.core import SignalR, StandardDetector

from geecs_bluesky.devices.ca.motor import CaMotor
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.devices.ca.snapshot import CaSnapshotReadable
from geecs_bluesky.devices.detector import STRICT_TRIGGER_INFO, GeecsDetector
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.namespace import (
    DeviceRoster,
    GeecsNamespace,
    identifier_name,
    looks_triggerable,
    motor_targets,
    python_type,
)


def row(name, *, settable=False, variabletype=None, choices="numeric", tolerance=None):
    return {
        "name": name,
        "settable": settable,
        "variabletype": variabletype,
        "choices": choices,
        "tolerance": tolerance,
        "units": "",
        "min": None,
        "max": None,
    }


ROSTER = DeviceRoster(
    experiment="TestExp",
    variables={
        "UC_TestCam": [
            row("MeanCounts"),
            row("MaxCounts"),
            row("Unsubscribed"),  # served? no: not subscribed, not settable
            row("exposure", settable=True, tolerance=0.0),  # numeric settable, DB tol 0
            row(
                "trigger", settable=True, choices="on,off"
            ),  # enum; collides with trigger()
            row("localsavingpath", settable=True, choices="path"),
            row("image", choices="image"),  # non-scalar
            row("bakground image", choices="image"),  # pushed only on demand
            row("processed image", choices="image"),
        ],
        "U_S1H": [
            row(
                "Current", settable=True, tolerance=0.05
            ),  # untyped numeric with tolerance
            row("Voltage"),  # subscribed readback
            row(
                "Enable_Output", settable=True, variabletype="choice", choices="on,off"
            ),
            row("Position.Axis 1"),  # not subscribed → not served
        ],
        "U_ImagesOnly": [row("image", choices="image")],
        "U DG645 ShotControl": [
            row("Trigger.Source", settable=True, choices="a,b"),
            row("Trigger.Rate", settable=True),
        ],
    },
    types={
        "UC_TestCam": "Point Grey Camera",
        "U_S1H": "Magnet PS",
        "U DG645 ShotControl": "DG645",
    },
    subscribed={
        "UC_TestCam": ["MeanCounts", "MaxCounts", "exposure"],  # exposure: settable too
        "U_S1H": ["Current", "Voltage"],
    },
)


# ------------------------------------------------------------------ rules
def test_identifier_name_keeps_geecs_spelling_when_valid() -> None:
    assert identifier_name("U_S1H") == "U_S1H"
    assert identifier_name("Current") == "Current"
    assert identifier_name("Position.Axis 1") == "position_axis_1"


def test_python_type_follows_effective_vartype() -> None:
    assert python_type(row("x")) is float  # choices=numeric
    assert python_type(row("x", choices="on,off")) is str  # enum → label string
    assert python_type(row("x", choices="path")) is str  # long string
    # variabletype wins over an option list (the gateway's served type today;
    # the 18 such Undulator rows are a DB fix — see geecs_core.db.variable_types)
    assert python_type(row("x", variabletype="numeric", choices="1,2,3")) is float
    assert python_type(row("x", choices="image")) is None  # non-scalar
    assert python_type(row("x", choices="1darray")) is None


def test_looks_triggerable_heuristic() -> None:
    assert looks_triggerable([row("trigger"), row("MeanCounts")], "Point Grey Camera")
    assert looks_triggerable([row("EnableTrigger")], "PicoscopeV2")
    assert not looks_triggerable([row("Current")], "Magnet PS")
    dg645 = [row("Trigger.Source"), row("Trigger.ExecuteSingleShot")]
    assert not looks_triggerable(dg645, "DG645")
    assert not looks_triggerable(dg645, "DG645 Delay Generator")  # token match
    assert looks_triggerable(dg645 + [row("acq_timestamp")], "DG645")  # DB row wins


# -------------------------------------------------------------- composition
def test_namespace_composes_the_existing_device_classes() -> None:
    ns = GeecsNamespace(ROSTER)
    assert set(ns.devices) == {"UC_TestCam", "U_S1H", "u_dg645_shotcontrol"}
    cam, magnet, box = ns["UC_TestCam"], ns["U_S1H"], ns["U DG645 ShotControl"]
    assert isinstance(cam, GeecsDetector)  # trigger-named variable → acquirer
    assert isinstance(magnet, CaSnapshotReadable)
    assert isinstance(box, CaSnapshotReadable)  # DG645 is a trigger *source*
    assert cam.name == "uc_testcam" and cam._geecs_device_name == "UC_TestCam"
    assert ns["u_dg645_shotcontrol"] is box and "uc_testcam" in ns and "nope" not in ns


def test_served_set_decides_which_children_exist() -> None:
    cam = GeecsNamespace(ROSTER)["UC_TestCam"]
    assert isinstance(cam.meancounts, SignalR) and isinstance(cam.maxcounts, SignalR)
    assert not hasattr(cam, "unsubscribed")  # not served
    assert not hasattr(cam, "image")  # non-scalar
    magnet = GeecsNamespace(ROSTER)["U_S1H"]
    assert not hasattr(magnet, "position_axis_1")


def test_settables_attach_as_movable_children_with_db_types() -> None:
    ns = GeecsNamespace(ROSTER)
    magnet, cam = ns["U_S1H"], ns["UC_TestCam"]
    assert isinstance(magnet.current, CaMotor)  # DB tolerance → convergence motor
    assert magnet.current._tolerance == pytest.approx(0.05)
    assert magnet.current.name == "u_s1h-current"  # named by the parent, safe_name
    assert isinstance(magnet.enable_output, CaSettable) and not isinstance(
        magnet.enable_output, CaMotor
    )
    # tol 0.0 → a plain setpoint, not a convergence motor (review #2)
    assert isinstance(cam.exposure, CaSettable) and not isinstance(
        cam.exposure, CaMotor
    )
    assert isinstance(cam.localsavingpath, CaSettable)  # no `save` row: not native_save
    # the settable's readback column header is the GEECS "Device Variable" form,
    # and the parent aggregates it (the s-file exporter reads top-level devices)
    assert magnet.current._column_headers == {"u_s1h-current-position": "U_S1H Current"}
    assert magnet._column_headers["u_s1h-current-position"] == "U_S1H Current"
    assert magnet._column_headers["u_s1h-voltage"] == "U_S1H Voltage"


def test_protocol_named_settable_binds_with_a_trailing_underscore() -> None:
    cam = GeecsNamespace(ROSTER)["UC_TestCam"]
    assert callable(cam.trigger) and cam.trigger.__func__ is GeecsDetector.trigger
    assert isinstance(cam.trigger_, CaSettable)


async def test_read_returns_the_subscribed_list_plus_shot_stamp() -> None:
    ns = GeecsNamespace(ROSTER)
    cam, magnet = ns["UC_TestCam"], ns["U_S1H"]
    await cam.connect(mock=True)
    await magnet.connect(mock=True)
    await cam.prepare(STRICT_TRIGGER_INFO)  # a StandardDetector reads once prepared
    # exposure is subscribed AND settable → its Movable child's readback is a
    # column of the detector (GeecsDetector.add_readables), like U_S1H.current
    assert set(await cam.read()) == {
        "uc_testcam-acq_timestamp",
        "uc_testcam-meancounts",
        "uc_testcam-maxcounts",
        "uc_testcam-exposure-readback",
    }
    assert set(await cam.describe()) == set(await cam.read())
    # Current is subscribed AND settable → its Movable child's readback is logged
    assert set(await magnet.read()) == {"u_s1h-current-position", "u_s1h-voltage"}
    assert hasattr(cam, "trigger") and not hasattr(magnet, "trigger")


def test_variable_and_resolve_accept_either_spelling() -> None:
    ns = GeecsNamespace(ROSTER)
    magnet = ns["U_S1H"]
    assert ns.variable("U_S1H", "Current") is magnet.current  # GEECS spelling
    assert ns.resolve("u_s1h:CURRENT") is magnet.current
    assert ns.resolve("U_S1H:voltage") is magnet.voltage
    assert ns.resolve("UC_TestCam:trigger") is ns["UC_TestCam"].trigger_
    assert ns.resolve("U_S1H") is magnet
    with pytest.raises(KeyError, match="no served scalar variable 'nope'"):
        ns.variable("U_S1H", "nope")
    with pytest.raises(KeyError, match="no device 'nope'"):
        ns["nope"]


def test_native_save_iff_the_db_lists_both_saving_controls() -> None:
    """§10.5: `save` + `localsavingpath` served → the detector owns them."""
    rows = list(ROSTER.variables["UC_TestCam"]) + [
        row("save", settable=True, choices="on,off")
    ]
    roster = DeviceRoster(
        experiment="TestExp",
        variables={"UC_TestCam": rows},
        types=ROSTER.types,
        subscribed=ROSTER.subscribed,
    )
    cam = GeecsNamespace(roster)["UC_TestCam"]
    assert isinstance(cam, StandardDetector) and cam.native_save
    # the data logic's own rw signals, not scan-settable children
    assert not isinstance(cam.save, CaSettable)
    assert not isinstance(cam.localsavingpath, CaSettable)
    assert cam.save.name == "uc_testcam-save"
    assert not GeecsNamespace(ROSTER)["UC_TestCam"].native_save  # no `save` row
    # get-only rows have no :SP (PV_CONTRACT.md §1): served, but not the
    # saving controls — they stay plain readables and the camera connects
    get_only = [row(n) for n in ("MeanCounts", "trigger", "save", "localsavingpath")]
    roster = DeviceRoster(
        experiment="TestExp",
        variables={"UC_TestCam": get_only},
        types=ROSTER.types,
        subscribed={"UC_TestCam": ["MeanCounts", "save", "localsavingpath"]},
    )
    cam = GeecsNamespace(roster)["UC_TestCam"]
    assert not cam.native_save
    assert isinstance(cam.save, SignalR) and not hasattr(cam.save, "_setpoint")


def test_roster_triggered_override_wins() -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables=ROSTER.variables,
        types=ROSTER.types,
        subscribed=ROSTER.subscribed,
        triggered={"U_S1H": True, "UC_TestCam": False},
    )
    ns = GeecsNamespace(roster)
    assert isinstance(ns["U_S1H"], GeecsDetector)
    assert isinstance(ns["UC_TestCam"], CaSnapshotReadable)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Trigger", "trigger"),  # case only
        ("Position.Axis 1", "Position Axis 1"),  # punctuation vs whitespace
        ("Wavelength (nm)", "wavelength_nm"),  # already-mangled spelling
    ],
)
def test_variables_normalising_to_one_attribute_are_refused(first, second) -> None:
    """safe_name is lossy (lowercase, punctuation runs → one underscore), so two
    served variables can land on one attribute/event key; refuse rather than
    silently drop one — as both gateways do on a PV collision after
    normalization (review #7 + codex P2)."""
    roster = DeviceRoster(
        experiment="TestExp",
        variables={"UC_X": [row(first), row(second)]},
        types={"UC_X": "Point Grey Camera"},
        subscribed={"UC_X": [first, second]},
    )
    with pytest.raises(GeecsConfigurationError, match="both normalise to"):
        GeecsNamespace(roster)


def test_settable_colliding_with_a_device_child_is_refused() -> None:
    """A settable named like a child the device class itself creates — the
    gateway liveness signal ``connected_status`` — would hide that child if it
    were merely renamed; raise instead (review N1).  (Variables colliding with
    each other are caught earlier, by the normalisation guard above.)"""
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "UC_X": [row("MeanCounts"), row("connected_status", settable=True)],
        },
        subscribed={"UC_X": ["MeanCounts"]},
        triggered={"UC_X": True},  # a GeecsDetector: it creates connected_status
    )
    with pytest.raises(GeecsConfigurationError, match="collides with the child"):
        GeecsNamespace(roster)


def test_name_collision_between_devices_is_loud() -> None:
    roster = DeviceRoster(
        experiment="TestExp",
        variables={
            "U Foo": [row("a", settable=True)],
            "U-Foo": [row("a", settable=True)],
        },
    )
    with pytest.raises(
        GeecsConfigurationError, match="both normalise to the name 'u_foo'"
    ):
        GeecsNamespace(roster)


def test_export_into_binds_names_and_refuses_shadowing() -> None:
    ns = GeecsNamespace(ROSTER)
    target: dict = {"RE": object()}
    assert ns.export_into(target) == ["UC_TestCam", "U_S1H", "u_dg645_shotcontrol"]
    assert target["U_S1H"] is ns["U_S1H"]
    with pytest.raises(GeecsConfigurationError, match="would shadow"):
        ns.export_into({"U_S1H": object()})


# ----------------------------------------------------------------- DB path
class _FakeDb:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def _q(self, name, value):
        self.calls.append(name)
        if self.fail:
            raise ConnectionError("db down")
        return value

    def get_experiment_device_variables(self, experiment, *, enabled_only=True):
        return self._q("variables", ROSTER.variables)

    def get_experiment_device_types(self, experiment, *, enabled_only=True):
        return self._q("types", ROSTER.types)

    def get_subscribed_variables(self, experiment, *, enabled_only=True):
        return self._q("subscribed", ROSTER.subscribed)

    def get_experiment_devices(self, experiment, *, enabled_only=True):
        return self._q(
            "endpoints",
            {device: ("192.168.6.100", 5000) for device in ROSTER.variables},
        )


def test_from_experiment_reuses_the_db_runtime_providers() -> None:
    db = _FakeDb()
    ns = GeecsNamespace.from_experiment("TestExp", geecs_db=db)
    assert (
        ns.roster.served
        == {  # GeecsDbServedSetProvider's rule: subscribed ∪ settable
            "UC_TestCam": {
                "MeanCounts",
                "MaxCounts",
                "exposure",
                "trigger",
                "localsavingpath",
            },
            "U_S1H": {"Current", "Voltage", "Enable_Output"},
            "U DG645 ShotControl": {"Trigger.Source", "Trigger.Rate"},
        }
    )
    assert ns.roster.types["UC_TestCam"] == "Point Grey Camera"
    assert set(ns.devices) == {"UC_TestCam", "U_S1H", "u_dg645_shotcontrol"}
    assert "variables" in db.calls and "subscribed" in db.calls and "types" in db.calls


def test_db_failure_at_build_is_loud_not_empty() -> None:
    with pytest.raises(
        GeecsConfigurationError, match="could not load the 'TestExp' device roster"
    ):
        GeecsNamespace.from_experiment("TestExp", geecs_db=_FakeDb(fail=True))


def test_reserved_device_attributes_pin_the_detector_class() -> None:
    """The client seam's frozen collision set is exactly the detector's public names."""
    from geecs_bluesky.devices.detector import GeecsDetector
    from geecs_bluesky.utils import RESERVED_DEVICE_ATTRIBUTES, settable_attribute

    public = {n for n in dir(GeecsDetector) if not n.startswith("_")}
    assert RESERVED_DEVICE_ATTRIBUTES == public
    assert settable_attribute("trigger") == "trigger_"
    assert settable_attribute("Current") == "current"
    assert settable_attribute("Position.Axis 1") == "position_axis_1"


# --------------------------------------------------------------- #806 rule
def _roster_on(host: str | None) -> DeviceRoster:
    import dataclasses

    endpoints = {} if host is None else {"UC_TestCam": host, "U_ImagesOnly": host}
    return dataclasses.replace(ROSTER, endpoints=endpoints)


def test_camera_on_a_plugin_host_is_plugin_backed() -> None:
    """DB image variable + endpoint in the file-plugin host list → stock ADHDFDataLogic."""
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider

    ns = GeecsNamespace(
        _roster_on("192.168.6.100"),
        path_provider=GeecsScanPathProvider(),
        file_plugin_hosts={"192.168.6.100"},
    )
    cam = ns.devices["UC_TestCam"]
    assert cam.plugin_backed
    assert cam.hdf.capture.source == "pva://testexp:uc_testcam:image:hdf1:Capture_RBV"
    # Only the primary image variable is captured: the DB's other image
    # variables are pushed only when an operation produces them.
    assert not hasattr(cam, "hdf_bakground_image") and len(cam._hdf_ios) == 1
    # The plugin's IO is never a telemetry object (only the detector's scalars are).
    assert cam.hdf not in ns.telemetry()
    assert all(not hasattr(obj, "num_captured") for obj in ns.telemetry())


def test_camera_elsewhere_keeps_labview_native_saving() -> None:
    """Not on a plugin host (or no host list, or no path provider): no plugin child."""
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider

    for kwargs in (
        {
            "path_provider": GeecsScanPathProvider(),
            "file_plugin_hosts": {"192.168.6.7"},
        },
        {"path_provider": GeecsScanPathProvider(), "file_plugin_hosts": None},
        {"path_provider": None, "file_plugin_hosts": {"192.168.6.100"}},
    ):
        ns = GeecsNamespace(_roster_on("192.168.6.100"), **kwargs)
        assert not ns.devices["UC_TestCam"].plugin_backed
        assert not hasattr(ns.devices["UC_TestCam"], "hdf")


def test_file_plugin_hosts_default_reads_the_config(monkeypatch) -> None:
    """The default host list is config.ini's; the test asserts the seam, not the lab."""
    import geecs_bluesky.namespace as namespace_module

    monkeypatch.setattr(
        namespace_module, "_hosts_from_config", lambda: {"192.168.6.100"}
    )
    from geecs_bluesky.plans.claim_scan import GeecsScanPathProvider

    ns = GeecsNamespace(
        _roster_on("192.168.6.100"), path_provider=GeecsScanPathProvider()
    )
    assert ns.devices["UC_TestCam"].plugin_backed


# ------------------------------------------------- measured drain offsets


def _offset(device) -> float:
    """The detector's seeded drain offset, read off the soft signal."""
    import asyncio

    async def go() -> float:
        await device.drain_offset.connect()
        return await device.drain_offset.get_value()

    return asyncio.run(go())


def test_measured_drain_offsets_reach_the_detectors() -> None:
    """The calibration's numbers are seeded into the config signal at build.

    This is the delivery path of ``shot_offsets.yaml``: the resolver's
    document becomes this mapping, which becomes each detector's
    ``drain_offset``, which rides in every descriptor and is what the s-file
    join corrects stamps by (``03`` §4.F).  A namespace that dropped the
    mapping would leave every device at 0.0 — exactly the state the
    calibration exists to end — with nothing failing to say so.
    """
    ns = GeecsNamespace(ROSTER, drain_offsets={"uc_testcam": 0.036})
    assert _offset(ns["UC_TestCam"]) == pytest.approx(0.036)


def test_a_device_without_a_measured_offset_keeps_zero() -> None:
    """Absent from the document means "stamps with the reference", i.e. 0.0."""
    ns = GeecsNamespace(ROSTER, drain_offsets={"uc_testcam": 0.036})
    other = [
        d
        for d in ns.devices.values()
        if isinstance(d, GeecsDetector) and d.name != "uc_testcam"
    ]
    for device in other:
        assert _offset(device) == 0.0


def test_an_offset_naming_no_detector_warns_rather_than_failing(caplog) -> None:
    """A stale calibration (renamed or retired camera) must be visible.

    It cannot be an error — the worker has to come up — but it must not be
    silent either: the device it meant to correct is left at 0.0 and its
    rows misjoin once the windows tighten, with nothing else to point at.
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="geecs_bluesky.namespace"):
        GeecsNamespace(ROSTER, drain_offsets={"uc_camera_that_left": 0.05})
    assert any(
        "uc_camera_that_left" in record.message and "stale" in record.message
        for record in caplog.records
    )


# ------------------------------------------------------------------ pseudos


def _catalog(**entries):
    from geecs_schemas.scan_variables import ScanVariables

    return ScanVariables.model_validate(
        {"schema_version": 1, "variables": entries}
    ).variables


def _magnet_roster() -> DeviceRoster:
    return DeviceRoster(
        experiment="TestExp",
        variables={
            "U_S3H": [row("Current", settable=True, tolerance=0.0), row("Voltage")],
            "U_S4H": [row("Current", settable=True, tolerance=0.02)],
            "U_S1H": [row("Current", settable=True, tolerance=0.05)],
        },
        types={"U_S3H": "Magnet PS", "U_S4H": "Magnet PS", "U_S1H": "Magnet PS"},
        subscribed={
            "U_S3H": ["Current", "Voltage"],
            "U_S4H": ["Current"],
            "U_S1H": ["Current"],
        },
    )


BUMP = {
    "kind": "pseudo",
    "mode": "relative",
    "targets": [
        {"target": "U_S3H:Current", "forward": "x"},
        {"target": "U_S4H:Current", "forward": "x * -2"},
    ],
}


def test_add_pseudos_binds_catalog_pseudos_over_the_bound_children() -> None:
    from geecs_bluesky.devices.ca.pseudo import (
        DEFAULT_AGREEMENT_TOLERANCE,
        CaPseudoPositioner,
    )

    ns = GeecsNamespace(_magnet_roster(), file_plugin_hosts=None)
    bound = ns.add_pseudos(
        _catalog(
            ALine_e_beam_angle_offset_x=BUMP,
            S3H={"target": "U_S3H:Current", "kind": "motor"},  # plain: ignored here
        )
    )
    assert bound == ["ALine_e_beam_angle_offset_x"]
    pseudo = ns["ALine_e_beam_angle_offset_x"]
    assert isinstance(pseudo, CaPseudoPositioner)
    assert pseudo.name == "aline_e_beam_angle_offset_x"
    assert pseudo.relative is True
    # the components ARE the roster's Movable children (one object, one offset)
    assert pseudo._components[0] is ns.variable("U_S3H", "Current")
    assert pseudo._components[1] is ns.variable("U_S4H", "Current")
    # agreement tolerance: the DB tolerance where set, the default where 0
    assert pseudo._tolerances == [DEFAULT_AGREEMENT_TOLERANCE, 0.02]
    assert pseudo._geecs_namespace_member is True
    assert (
        "ALine_e_beam_angle_offset_x" in ns
        and ns["aline_e_beam_angle_offset_x"] is pseudo
    )
    # exported like a device; not in the telemetry baseline (its components are)
    exported: dict = {}
    assert "ALine_e_beam_angle_offset_x" in ns.export_into(exported)
    assert pseudo not in ns.telemetry()


def test_add_pseudos_skips_a_broken_entry_loudly_and_keeps_the_rest(caplog) -> None:
    ns = GeecsNamespace(_magnet_roster(), file_plugin_hosts=None)
    with caplog.at_level("ERROR", logger="geecs_bluesky.namespace"):
        bound = ns.add_pseudos(
            _catalog(
                unserved={
                    "kind": "pseudo",
                    "mode": "absolute",
                    "targets": [{"target": "U_Nope:Current", "forward": "x"}],
                },
                not_settable={
                    "kind": "pseudo",
                    "mode": "absolute",
                    "targets": [{"target": "U_S3H:Voltage", "forward": "x"}],
                },
                bad_relative={
                    "kind": "pseudo",
                    "mode": "relative",
                    "targets": [{"target": "U_S3H:Current", "forward": "x + 1"}],
                },
                U_S1H={  # collides with a device binding
                    "kind": "pseudo",
                    "mode": "absolute",
                    "targets": [{"target": "U_S3H:Current", "forward": "x"}],
                },
                good=BUMP,
            )
        )
    assert bound == ["good"]
    assert "unserved" not in ns and "good" in ns
    messages = [r.getMessage() for r in caplog.records]
    assert any("'unserved' not registered" in m and "U_Nope" in m for m in messages)
    assert any(
        "'not_settable' not registered" in m and "not settable" in m for m in messages
    )
    assert any(
        "'bad_relative' not registered" in m and "not 0 at 0" in m for m in messages
    )
    assert any("'U_S1H' not registered" in m and "already bound" in m for m in messages)
    assert ns["U_S1H"].current is ns.variable("U_S1H", "Current")  # the device survived


def test_add_pseudos_refuses_case_clashes_and_plan_names(caplog) -> None:
    ns = GeecsNamespace(_magnet_roster(), file_plugin_hosts=None)
    with caplog.at_level("ERROR", logger="geecs_bluesky.namespace"):
        bound = ns.add_pseudos(_catalog(u_s1h=BUMP, count=BUMP, RE=BUMP, bump=BUMP))
    assert bound == ["bump"]
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "'u_s1h' not registered" in m and "already bound to 'U_S1H'" in m
        for m in messages
    )
    assert any("'count' not registered" in m and "plan name" in m for m in messages)
    assert any("'RE' not registered" in m and "plan name" in m for m in messages)
    assert ns.get_settable("u_s1h", "Current") is ns["U_S1H"].current  # lookups intact


def test_namespace_pseudo_scans_through_connect_on_demand() -> None:
    """A pseudo touched by a plan connects its own components: the telemetry
    connect at environment open may have left one out (run_engine.install_telemetry
    drops members that fail to connect)."""
    import bluesky.plan_stubs as bps
    import bluesky.plans as bp
    from bluesky import RunEngine

    from ophyd_async.core import callback_on_mock_put, set_mock_value

    from geecs_bluesky.preprocessors import install_connect_on_demand

    ns = GeecsNamespace(_magnet_roster(), file_plugin_hosts=None)
    ns.add_pseudos(_catalog(ALine_e_beam_angle_offset_x=BUMP))
    bump = ns["ALine_e_beam_angle_offset_x"]
    RE = RunEngine()
    install_connect_on_demand(RE, mock=True)  # nothing connected up front
    values: list[float] = []

    def plan():
        # stage() reads the components' readbacks: it only works because the
        # pseudo's connect brought its components along
        yield from bps.stage(bump, wait=True)
        for comp in bump._components:  # mock backends exist now: readbacks follow puts
            readback = getattr(comp, comp._readback_attr_name)
            callback_on_mock_put(
                comp._setpoint,
                lambda value, *, readback=readback, **kw: set_mock_value(
                    readback, value
                ),
            )
        yield from bps.unstage(bump, wait=True)
        yield from bp.scan([], bump, -1.0, 1.0, 3)

    RE(
        plan(),
        lambda name, doc: values.append(
            doc["data"]["aline_e_beam_angle_offset_x-readback"]
        )
        if name == "event"
        else None,
    )
    assert values == pytest.approx([-1.0, 0.0, 1.0])


# -------------------------------------------------------- kind: motor opt-in


def test_motor_targets_reads_plain_kind_motor_entries_only() -> None:
    catalog = _catalog(
        S3H={"target": "U_S3H:Current", "kind": "motor"},
        Gas={"target": "U_HP_Daq:AnalogOutput.Channel 1", "kind": "setpoint"},
        Default={"target": "U_S4H:Current"},  # kind defaults to setpoint
        bump=BUMP,  # a pseudo's components are not opted in here
    )
    assert motor_targets(catalog) == {"u_s3h:current"}


def test_catalog_kind_motor_binds_a_motor_where_the_db_tolerance_is_zero(
    caplog,
) -> None:
    from geecs_bluesky.devices.ca.motor import DEFAULT_TOLERANCE

    with caplog.at_level("WARNING", logger="geecs_bluesky.namespace"):
        ns = GeecsNamespace(
            _magnet_roster(), file_plugin_hosts=None, motor_targets={"U_S3H:Current"}
        )
    s3h = ns.variable("U_S3H", "Current")
    assert isinstance(s3h, CaMotor) and s3h._tolerance == DEFAULT_TOLERANCE
    assert any(
        "U_S3H:Current is a catalog 'kind: motor' but its DB tolerance is 0.0"
        in r.getMessage()
        for r in caplog.records
    )
    # a DB tolerance still wins its own value; an un-opted 0 stays a plain setpoint
    assert isinstance(ns.variable("U_S4H", "Current"), CaMotor)
    assert ns.variable("U_S4H", "Current")._tolerance == 0.02
    plain = GeecsNamespace(_magnet_roster(), file_plugin_hosts=None)
    assert type(plain.variable("U_S3H", "Current")) is CaSettable


def test_catalog_kind_setpoint_never_downgrades_a_db_motor() -> None:
    ns = GeecsNamespace(_magnet_roster(), file_plugin_hosts=None, motor_targets=set())
    assert isinstance(ns.variable("U_S1H", "Current"), CaMotor)
