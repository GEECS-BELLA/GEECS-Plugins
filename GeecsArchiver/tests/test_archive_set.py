"""The rule, on hand-built DB rows."""

from geecs_schemas import ArchivePolicy, DerivedChannels

from geecs_archiver.archive_set import (
    ArchiveCandidate,
    is_timestamp_variable,
    Sampling,
    classify,
    derive_candidates,
    is_excluded,
    sampling_for,
)


def row(name, *, settable=False, variabletype=None, choices=None):
    return {
        "name": name,
        "settable": settable,
        "variabletype": variabletype,
        "choices": choices,
    }


ENDPOINTS = {
    "U_S1H": ("10.0.0.1", 1),
    "UC_Cam3": ("10.0.0.2", 2),
    "U_Idle": ("10.0.0.3", 3),
    "U_Img": ("10.0.0.4", 4),
}
VAR_MAP = {
    "U_S1H": [
        row("Current", settable=True),
        row("Voltage"),
        row("systimestamp"),
        row("acq_timestamp"),
        row("Current"),
    ],
    "UC_Cam3": [
        row("centroidx"),
        row("timestamp"),
        row("localsavingpath", settable=True, choices="path"),
        row("save", settable=True, variabletype="choice", choices="on,off"),
        row("exposure", settable=True),
    ],
    "U_Idle": [row("Nothing")],
    "U_Img": [row("image", variabletype="choice", choices="image")],
}
SUB_MAP = {
    "U_S1H": ["Current", "Voltage", "systimestamp", "acq_timestamp"],
    "UC_Cam3": ["centroidx", "localsavingpath", "timestamp"],
}


def pvs(policy=None, derived=None):
    return [
        c.pv
        for c in derive_candidates(
            "Undulator",
            ENDPOINTS,
            VAR_MAP,
            SUB_MAP,
            policy=policy or ArchivePolicy(),
            derived=derived,
        )
    ]


def test_monitored_readbacks_and_settable_setpoints_with_normalized_names():
    out = pvs()
    assert "undulator:u_s1h:current" in out
    assert "undulator:u_s1h:current:SP" in out
    assert "undulator:u_s1h:voltage" in out
    assert "undulator:u_s1h:voltage:SP" not in out  # not settable
    assert "undulator:uc_cam3:centroidx" in out


def test_settable_but_unmonitored_gives_only_the_setpoint():
    out = pvs()
    assert "undulator:uc_cam3:exposure:SP" in out
    assert "undulator:uc_cam3:exposure" not in out
    assert "undulator:uc_cam3:save:SP" in out


def test_timestamps_paths_and_images_are_never_archived():
    out = pvs()
    assert not any(
        "timestamp" in pv for pv in out
    )  # acq_timestamp, systimestamp and a device's own 'timestamp'
    assert not any("localsavingpath" in pv for pv in out)
    assert not any(pv.startswith("undulator:u_img:") for pv in out)


def test_status_pv_follows_the_gateway_rule():
    out = pvs()
    assert "undulator:u_s1h:connected" in out
    assert (
        "undulator:uc_cam3:connected" in out
    )  # serves a path variable → the device exists
    assert (
        "undulator:u_idle:connected" not in out
    )  # exposes nothing → the gateway skips it
    assert "undulator:u_img:connected" not in out  # image-only → nothing on CA
    assert "undulator:u_s1h:connected" not in pvs(ArchivePolicy(include_status=False))


def test_setpoints_can_be_switched_off():
    out = pvs(ArchivePolicy(include_setpoints=False))
    assert not any(pv.endswith(":SP") for pv in out)
    assert "undulator:u_s1h:current" in out


def test_derived_channels_are_included_and_switchable():
    derived = DerivedChannels.model_validate(
        {
            "derived_channels": [
                {
                    "device": "TargetChamberPressure",
                    "variable": "Pressure",
                    "expression": "v",
                    "inputs": [
                        {"symbol": "v", "device": "U_S1H", "variable": "Current"}
                    ],
                }
            ]
        }
    )
    assert "undulator:targetchamberpressure:pressure" in pvs(derived=derived)
    assert "undulator:targetchamberpressure:pressure" not in pvs(
        ArchivePolicy(include_derived=False), derived=derived
    )


def test_exclude_globs_are_case_folded_and_applied_last():
    policy = ArchivePolicy(exclude=["UNDULATOR:UC_CAM3:*", "*:connected"])
    out = pvs(policy)
    assert not any(pv.startswith("undulator:uc_cam3:") for pv in out)
    assert not any(pv.endswith(":connected") for pv in out)
    assert "undulator:u_s1h:current" in out
    assert is_excluded("undulator:uc_cam3:centroidx", policy)


def test_duplicate_rows_collapse_and_output_is_sorted():
    out = pvs()
    assert out == sorted(out)
    assert out.count("undulator:u_s1h:current") == 1


def test_candidates_carry_provenance():
    cands = derive_candidates(
        "Undulator", ENDPOINTS, VAR_MAP, SUB_MAP, policy=ArchivePolicy()
    )
    by_pv = {c.pv: c for c in cands}
    assert by_pv["undulator:u_s1h:current:SP"] == ArchiveCandidate(
        "undulator:u_s1h:current:SP", "U_S1H", "Current", "setpoint", "float"
    )
    assert by_pv["undulator:uc_cam3:save:SP"].dtype == "enum"
    assert by_pv["undulator:u_s1h:connected"].kind == "status"


def test_classify_mirrors_the_shared_type_rule():
    assert classify(row("x")) == "float"
    assert classify(row("x", choices="path")) == "path"
    assert classify(row("x", variabletype="choice", choices="a,b")) == "enum"
    assert classify(row("x", choices="image")) is None
    assert classify(row("x", choices="1darray")) is None


def test_sampling_defaults_and_override_precedence():
    policy = ArchivePolicy(
        default_sampling_period=2.0,
        sampling_overrides=[
            {"match": "undulator:u_*", "policy": "Slow"},
            {"match": "undulator:u_s1h:*", "sampling_period": 0.5},
            {"match": "*:current:SP", "sampling_method": "SCAN"},
        ],
    )
    assert sampling_for("undulator:uc_cam3:centroidx", policy) == Sampling(
        2.0, "MONITOR", None
    )
    assert sampling_for("undulator:u_vac:pressure", policy) == Sampling(
        2.0, "MONITOR", "Slow"
    )
    assert sampling_for("undulator:u_s1h:voltage", policy) == Sampling(
        0.5, "MONITOR", "Slow"
    )
    assert sampling_for("undulator:u_s1h:current:SP", policy) == Sampling(
        0.5, "SCAN", "Slow"
    )


def test_sampling_request_body():
    assert Sampling(1.0, "MONITOR").request("a:b:c") == {
        "pv": "a:b:c",
        "samplingperiod": "1",
        "samplingmethod": "MONITOR",
    }
    assert Sampling(10.0, "MONITOR", "Slow").request("a:b:c")["policy"] == "Slow"


def test_timestamp_variable_rule():
    assert is_timestamp_variable("acq_timestamp") and is_timestamp_variable(
        "SysTimestamp"
    )
    assert is_timestamp_variable("timestamp") and is_timestamp_variable(
        "Shot Timestamp"
    )
    assert not is_timestamp_variable("timestamp_offset") and not is_timestamp_variable(
        "current"
    )
