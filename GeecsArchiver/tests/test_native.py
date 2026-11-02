"""The registry pass, on the installed family files and hand-written registries."""

from geecs_schemas import ArchivePolicy, NativeDevices, PanelDevices, load_families

from geecs_archiver import archive_set
from geecs_archiver.archive_set import build_archive_set, derive_candidates
from geecs_archiver.native import default_archive_roles, derive_native_candidates

FAMILIES = load_families()

MOTOR = {"U_ESP_JetXYZ": {"kind": "epics_motor", "axes": ["x", "y"]}}
CAMERA = {"UC_VisaEBeam6": {"kind": "aravis_camera", "trigger_source": "line0"}}
CAMERA_PANEL = {"UC_VisaEBeam6": {"family": "ad_camera_chain", "ioc": "uc-visaebeam6"}}
SUPPLY_PANEL = {
    "eztest1": {
        "family": "easydriver",
        "prefix": "undulator:eztest1:",
        "ioc": "eztest1",
    }
}

# Hand-written from the family files: what each device's archive set is.
MOTOR_PVS = sorted(
    f"undulator:u_esp_jetxyz:{axis}{field}"
    for axis in ("x", "y")
    for field in (".RBV", ".VAL", ".DMOV", ".HLS", ".LLS")
)
CAMERA_PVS = sorted(
    [
        "undulator:uc_visaebeam6:cam1:AcquireTime_RBV",
        "undulator:uc_visaebeam6:cam1:PixelFormat_RBV",
        "undulator:uc_visaebeam6:RUN:Lock",
        "undulator:uc_visaebeam6:cam1:AcquirePeriod_RBV",
        "undulator:uc_visaebeam6:cam1:FrameRate_RBV",
        "undulator:uc_visaebeam6:cam1:Gain_RBV",
        "undulator:uc_visaebeam6:cam1:TriggerMode_RBV",
        "undulator:uc_visaebeam6:cam1:TriggerSource_RBV",
        "undulator:uc_visaebeam6:cam1:Acquire_RBV",
        "undulator:uc_visaebeam6:cam1:MinX_RBV",
        "undulator:uc_visaebeam6:cam1:MinY_RBV",
        "undulator:uc_visaebeam6:cam1:SizeX_RBV",
        "undulator:uc_visaebeam6:cam1:SizeY_RBV",
        "undulator:uc_visaebeam6:cam1:AsynIO.CNCT",  # the kind's liveness
        "UC-VISAEBEAM6:HEARTBEAT",  # devIocStats, per IOC
    ]
)
SUPPLY_PVS = sorted(
    [
        "undulator:eztest1:current:SP",
        "undulator:eztest1:current",
        "undulator:eztest1:voltage",
        "undulator:eztest1:enable_output:SP",
        "undulator:eztest1:enable_output",
        "undulator:eztest1:fault",
        "undulator:eztest1:interlock",
        "EZTEST1:HEARTBEAT",
    ]
)


def native(registry=None, panels=None, *, policy=None, db=None, families=FAMILIES):
    return derive_native_candidates(
        "Undulator",
        NativeDevices.model_validate(
            {"devices": registry or {}}, context={"families": families}
        ),
        PanelDevices.model_validate(
            {"devices": panels or {}}, context={"families": families}
        ),
        policy=policy or ArchivePolicy(),
        db_devices=db,
        families=families,
    )


def pvs(result):
    return sorted(c.pv for c in result.candidates)


def test_a_motor_archives_each_axis_readback_setpoint_done_flag_and_switches():
    result = native(MOTOR)
    assert pvs(result) == MOTOR_PVS
    kinds = {c.pv: c.kind for c in result.candidates}
    assert kinds["undulator:u_esp_jetxyz:x.VAL"] == "setpoint"
    assert kinds["undulator:u_esp_jetxyz:x.RBV"] == "readback"
    assert {c.variable for c in result.candidates if c.pv.endswith("y.RBV")} == {
        "y:rbv"
    }


def test_a_camera_archives_its_settings_readbacks_lock_liveness_and_ioc_heartbeat():
    assert pvs(native(CAMERA, CAMERA_PANEL)) == CAMERA_PVS


def test_a_camera_without_panel_data_takes_its_kinds_family_and_no_heartbeat():
    expected = [pv for pv in CAMERA_PVS if not pv.endswith(":HEARTBEAT")]
    assert pvs(native(CAMERA)) == expected


def test_a_bench_supply_archives_readbacks_setpoints_and_fault_bits_at_its_prefix():
    result = native(panels=SUPPLY_PANEL, db=["U_S1H"])
    assert pvs(result) == SUPPLY_PVS
    kinds = {c.pv: c.kind for c in result.candidates}
    assert kinds["undulator:eztest1:current:SP"] == "setpoint"
    assert kinds["undulator:eztest1:enable_output:SP"] == "setpoint"
    assert kinds["undulator:eztest1:fault"] == "readback"


def test_include_setpoints_drops_records_that_are_their_own_write_target_only():
    result = native(
        MOTOR | CAMERA,
        CAMERA_PANEL | SUPPLY_PANEL,
        policy=ArchivePolicy(include_setpoints=False),
    )
    got = pvs(result)
    assert "undulator:u_esp_jetxyz:x.VAL" not in got
    assert "undulator:eztest1:current:SP" not in got
    # a setting's _RBV is the device's state, not the operator's intent: kept
    assert "undulator:uc_visaebeam6:cam1:AcquireTime_RBV" in got
    assert "undulator:u_esp_jetxyz:x.RBV" in got


def test_include_status_gates_the_liveness_record_and_the_heartbeats():
    got = pvs(
        native(
            CAMERA,
            CAMERA_PANEL | SUPPLY_PANEL,
            policy=ArchivePolicy(include_status=False),
        )
    )
    assert not [pv for pv in got if pv.endswith((":HEARTBEAT", "AsynIO.CNCT"))]
    assert "undulator:uc_visaebeam6:RUN:Lock" in got


def test_one_heartbeat_per_ioc_however_many_devices_it_serves():
    panels = SUPPLY_PANEL | {
        "eztest2": {
            "family": "easydriver",
            "prefix": "undulator:eztest2:",
            "ioc": "EZTEST1",
        }
    }
    got = pvs(native(panels=panels))
    assert got.count("EZTEST1:HEARTBEAT") == 1


def test_instance_opt_outs_and_opt_ins_change_the_familys_defaults():
    panels = {
        "U_ESP_JetXYZ": {
            "family": "motor_record",
            "no_archive": ["hls"],
            "archive": ["velo"],
            "omit": ["lls"],
        }
    }
    got = pvs(native(MOTOR, panels))
    assert "undulator:u_esp_jetxyz:x.HLS" not in got  # opted out
    assert "undulator:u_esp_jetxyz:x.LLS" not in got  # the device has none
    assert "undulator:u_esp_jetxyz:x.VELO" in got  # opted in
    assert "undulator:u_esp_jetxyz:y.RBV" in got


def test_arrays_images_and_commands_are_never_archived_even_opted_in():
    panels = {
        "UC_VisaEBeam6": {
            "family": "ad_camera_chain",
            "archive": ["profile_x", "image", "background_saving"],
        }
    }
    got = pvs(native(CAMERA, panels))
    assert not [
        pv for pv in got if "Profile" in pv or "JPG_URL" in pv or "SaveBackground" in pv
    ]


def test_exclude_globs_filter_registry_candidates_where_the_passes_meet():
    policy = ArchivePolicy(exclude=["undulator:u_esp_jetxyz:*.dmov"])
    result = native(MOTOR, policy=policy)
    got = [
        c.pv
        for c in derive_candidates(
            "Undulator", {}, {}, {}, policy=policy, native=result.candidates
        )
    ]
    assert "undulator:u_esp_jetxyz:x.DMOV" not in got
    assert "undulator:u_esp_jetxyz:x.RBV" in got


def test_every_candidate_carries_its_source():
    rows = {
        "U_S1H": [
            {
                "name": "Current",
                "settable": False,
                "variabletype": None,
                "choices": None,
            }
        ]
    }
    got = derive_candidates(
        "Undulator",
        {"U_S1H": ("h", 1)},
        rows,
        {"U_S1H": ["Current"]},
        policy=ArchivePolicy(),
        native=native(MOTOR).candidates,
    )
    sources = {c.pv: c.source for c in got}
    assert sources["undulator:u_s1h:current"] == "db"
    assert sources["undulator:cagateway:heartbeat"] == "db"
    assert sources["undulator:u_esp_jetxyz:x.RBV"] == "registry"


def test_a_registry_device_the_db_also_owns_is_reported_not_doubled():
    result = native(MOTOR | CAMERA, db=["U_ESP_JETXYZ", "U_S1H"])
    assert len(result.conflicts) == 1 and "U_ESP_JetXYZ" in result.conflicts[0]
    assert not [c for c in result.candidates if c.device == "U_ESP_JetXYZ"]
    assert "undulator:uc_visaebeam6:cam1:Gain_RBV" in pvs(result)


def test_a_bench_device_the_db_owns_is_the_gateways_and_skipped():
    panels = {"U_S1H": {"family": "easydriver", "prefix": "undulator:bench1:"}}
    result = native(panels=panels, db=["U_S1H"])
    assert result.candidates == [] and result.conflicts == []
    assert "U_S1H" in result.skipped[0]


def test_without_the_db_nothing_is_checked_and_it_says_so():
    result = native(MOTOR, panels={"U_S1H": {"family": "easydriver"}})
    assert result.db_checked is False and result.conflicts == []
    assert "undulator:u_s1h:current" in pvs(result)


def unmarked(template):
    family = FAMILIES[template]
    return family.model_copy(
        update={
            "roles": [r.model_copy(update={"archive": False}) for r in family.roles]
        }
    )


def test_a_motor_family_marking_nothing_falls_back_to_the_kinds_rule():
    families = dict(FAMILIES) | {"motor_record": unmarked("motor_record")}
    assert pvs(native(MOTOR, families=families)) == MOTOR_PVS


def test_a_supply_family_marking_nothing_takes_readbacks_setpoints_and_bits():
    family = unmarked("easydriver")
    assert default_archive_roles(family) == {
        "current_sp",
        "current",
        "voltage",
        "output_sp",
        "output",
        "fault",
        "interlock",
    }


def test_a_detector_family_marking_nothing_takes_settings_readbacks_and_the_lock():
    roles = default_archive_roles(unmarked("ad_camera_chain"))
    assert {"exposure", "gain", "acquire", "lock"} <= roles
    assert not {"cx", "rate", "frame_rate", "image", "pixel_format"} & roles
    roles = default_archive_roles(unmarked("scope_contract"))
    assert {"samples", "ch_a_offset"} <= roles
    assert not {"ch_a_mean", "data", "connected"} & roles


def test_build_archive_set_checks_the_registry_against_the_db_roster(monkeypatch):
    from geecs_core.db import geecs_db

    monkeypatch.setattr(
        geecs_db.GeecsDb,
        "get_experiment_devices",
        staticmethod(lambda e, enabled_only=True: {"UC_VisaEBeam6": ("h", 1)}),
    )
    monkeypatch.setattr(
        geecs_db.GeecsDb,
        "get_experiment_device_variables",
        staticmethod(lambda e, enabled_only=True: {"UC_VisaEBeam6": []}),
    )
    monkeypatch.setattr(
        geecs_db.GeecsDb,
        "get_subscribed_variables",
        staticmethod(lambda e, enabled_only=True: {}),
    )
    result = build_archive_set(
        "Undulator",
        policy=ArchivePolicy(),
        registry=NativeDevices.model_validate({"devices": CAMERA}),
    )
    assert result.registry.conflicts and result.by_source("registry") == []


def test_build_archive_set_without_the_db_never_opens_it(monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the DB was read")

    from geecs_core.db import geecs_db

    monkeypatch.setattr(
        geecs_db.GeecsDb, "get_experiment_devices", staticmethod(refuse)
    )
    result = build_archive_set(
        "Undulator",
        policy=ArchivePolicy(),
        registry=NativeDevices.model_validate({"devices": MOTOR}),
        use_db=False,
    )
    assert [c.pv for c in result.candidates] == MOTOR_PVS
    assert result.registry.db_checked is False


def test_no_db_set_carries_none_of_the_gateways_pvs():
    result = build_archive_set(
        "Undulator",
        policy=ArchivePolicy(),
        registry=NativeDevices.model_validate({"devices": MOTOR}),
        use_db=False,
    )
    assert {c.source for c in result.candidates} == {"registry"}
    assert archive_set.GATEWAY_DEVICE not in {c.device for c in result.candidates}


def test_a_registry_device_it_cannot_resolve_is_drift_not_a_silent_drop():
    # panel data naming a family of another kind: PanelDevices cannot catch it
    panels = {"UC_VisaEBeam6": {"family": "motor_record", "axes": ["x"]}}
    result = native(CAMERA, panels)
    assert result.candidates == [] and result.drift
    assert "UC_VisaEBeam6" in result.unresolved[0]


def test_a_registry_kind_defaults_to_the_shared_tables_family_whatever_else_is_installed():
    camera = FAMILIES["ad_camera_chain"]
    families = dict(FAMILIES) | {
        "another_chain": camera.model_copy(update={"template": "another_chain"})
    }
    assert pvs(native(CAMERA, families=families)) == [
        pv for pv in CAMERA_PVS if not pv.endswith(":HEARTBEAT")
    ]


def _gain_only_chain():
    """A second camera family carrying one archived role: which family won is visible in the PVs."""
    camera = FAMILIES["ad_camera_chain"]
    gain = [r for r in camera.roles if r.suffix == "cam1:Gain_RBV"]
    assert gain, "the installed camera family lost its Gain_RBV role"
    return dict(FAMILIES) | {
        "gain_chain": camera.model_copy(
            update={"template": "gain_chain", "roles": gain, "actions": {}}
        )
    }


def test_a_registry_template_picks_the_family_over_the_kinds_default():
    registry = {"UC_VisaEBeam6": CAMERA["UC_VisaEBeam6"] | {"template": "gain_chain"}}
    result = native(registry, families=_gain_only_chain())
    assert pvs(result) == [
        "undulator:uc_visaebeam6:cam1:AsynIO.CNCT",
        "undulator:uc_visaebeam6:cam1:Gain_RBV",
    ]


def test_panel_data_naming_another_family_than_the_registrys_is_unresolved():
    registry = {"UC_VisaEBeam6": CAMERA["UC_VisaEBeam6"] | {"template": "gain_chain"}}
    result = native(registry, CAMERA_PANEL, families=_gain_only_chain())
    assert result.candidates == [] and result.drift
    assert "one family per device" in result.unresolved[0]


def test_panel_data_moving_a_registry_devices_prefix_is_unresolved():
    panels = {"UC_VisaEBeam6": CAMERA_PANEL["UC_VisaEBeam6"] | {"prefix": "bench:cam:"}}
    result = native(CAMERA, panels)
    assert result.candidates == [] and result.drift
    assert "prefix" in result.unresolved[0]
