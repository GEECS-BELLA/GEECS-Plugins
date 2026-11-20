"""The demo file and its script: every kind scripted, the windmill's settings moving."""

from pathlib import Path

from geecs_interlocks.config import load_interlocks
from geecs_interlocks.demo import DemoScript
from geecs_interlocks.engine import device_variables, external_ports

DEMO = Path(__file__).parent.parent / "examples" / "demo_interlocks.yaml"


def test_the_demo_file_holds_every_kind():
    interlocks, _ = load_interlocks(DEMO)
    kinds = {c.kind for i in interlocks.interlocks for c in i.conditions}
    assert kinds == {"threshold", "alignment", "stale", "external", "config_change"}
    assert any(len(i.conditions) > 1 for i in interlocks.interlocks)
    assert external_ports(interlocks) == {5050}


def test_every_watched_variable_is_scripted():
    interlocks, _ = load_interlocks(DEMO)
    script = DemoScript.from_interlocks(interlocks)
    for device, variables in device_variables(interlocks).items():
        assert set(variables) <= script.devices[device]


def test_the_windmill_settings_move_and_come_back():
    interlocks, _ = load_interlocks(DEMO)
    script = DemoScript.from_interlocks(interlocks)
    positions = {
        script.values("DEMO_Windmill", float(t))["Position"] for t in range(200)
    }
    assert len(positions) == 2
    assert script.values("DEMO_Windmill", 0.0)["Position"] == min(positions)
