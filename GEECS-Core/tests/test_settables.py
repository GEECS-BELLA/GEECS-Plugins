"""The numeric-settables list: the filter and the alias-first order every picker shares."""

from __future__ import annotations

from geecs_core.db.settables import NumericSettable, numeric_settables


def _row(name, *, settable=True, vartype="numeric", choices=None, alias="", units=""):
    return {
        "name": name,
        "settable": settable,
        "variabletype": vartype,
        "choices": choices,
        "alias": alias,
        "units": units,
        "min": None,
        "max": None,
    }


def test_keeps_numeric_settables_only() -> None:
    rows = {
        "U_A": [
            _row("Current", units="A"),
            _row("Readback", settable=False),
            _row("Enable", vartype="choice", choices="on,off"),
            _row("Label", vartype="string"),
            _row(
                "Image", vartype="", choices="image"
            ),  # descriptor wins over a blank type
            _row("Gap", vartype="", choices=None),  # blank type, no options → numeric
            _row("  ", units="A"),  # a blank name is not a variable
        ],
    }
    assert [s.name for s in numeric_settables(rows)] == ["U_A:Current", "U_A:Gap"]


def test_aliased_first_then_canonical_order() -> None:
    rows = {
        "U_Zaber": [_row("Position", alias="Zed stage")],
        "U_Jet": [
            _row("Position.Axis 3", alias=" Jet_Z (mm) ", units=" mm "),
            _row("Position.Axis 1"),
        ],
        "U_S1H": [_row("Current")],
        "U_EMQ1": [_row("Current")],
    }
    out = numeric_settables(rows)
    assert [s.alias or s.name for s in out] == [
        "Jet_Z (mm)",
        "Zed stage",  # aliased, alphabetical by alias
        "U_EMQ1:Current",
        "U_Jet:Position.Axis 1",
        "U_S1H:Current",  # the rest by canonical name
    ]
    assert out[0] == NumericSettable(
        name="U_Jet:Position.Axis 3",
        device="U_Jet",
        variable="Position.Axis 3",
        alias="Jet_Z (mm)",
        units="mm",
    )
