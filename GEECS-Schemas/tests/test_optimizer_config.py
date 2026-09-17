"""Optimization documents use GEST directly and validate reference dependencies."""

from copy import deepcopy

import pytest
from gest_api.vocs import VOCS
from geecs_schemas import OptimizerConfig, optimizer_required_devices
from geecs_schemas.optimizer_config import MEASUREMENT_MATH
from geecs_schemas.restricted_expr import compile_expression, ExpressionWhitelistError


def document():
    return dict(
        vocs=dict(
            variables={"Motor:Current": [-1, 1]}, objectives={"brightness": "MAXIMIZE"}
        ),
        measurements=dict(
            cam=dict(diagnostic="Camera"), charge=dict(signal="ICT:Charge")
        ),
        derived=dict(brightness="cam.image_total / charge"),
        generator=dict(name="random"),
    )


def test_gest_identity_roundtrip_and_no_input_mutation():
    cfg = OptimizerConfig(**document())
    assert type(cfg.vocs) is VOCS
    dumped = cfg.model_dump(mode="json")
    before = deepcopy(dumped)
    assert OptimizerConfig.model_validate(dumped).model_dump(mode="json") == before
    assert dumped == before
    assert OptimizerConfig.model_json_schema()["properties"]["vocs"]["type"] == "object"
    assert optimizer_required_devices(cfg, {"Camera": "CameraDevice"}) == {
        "CameraDevice",
        "ICT",
    }


@pytest.mark.parametrize(
    "expression",
    [
        "unknown + charge",
        "later + charge",
        "cam.__class__()",
        "charge[0]",
        "__import__('os')",
    ],
)
def test_unknown_forward_or_unsafe_expression_refused(expression):
    data = document()
    data["derived"] = dict(brightness=expression, later="charge")
    with pytest.raises(ValueError):
        OptimizerConfig(**data)


def test_dotted_names_never_access_objects_or_collide():
    expr = compile_expression(
        "cam.x + _geecs_symbol_0", {"cam.x", "_geecs_symbol_0"}, MEASUREMENT_MATH
    )
    assert expr.evaluate({"cam.x": 3, "_geecs_symbol_0": 4}) == 7
    with pytest.raises(ExpressionWhitelistError):
        compile_expression("cam.x.__class__", {"cam.x"}, MEASUREMENT_MATH)


def test_legacy_document_refused():
    """The refusal names the action to take, not a document to go read."""
    with pytest.raises(ValueError, match="legacy optimizer config is not loadable"):
        OptimizerConfig.model_validate({"evaluator": {}})
    with pytest.raises(ValueError, match="schema_version 1"):
        OptimizerConfig.model_validate({"device_requirements": {}})


def test_diagnostic_overrides_cannot_change_device():
    data = document()
    data["measurements"]["cam"]["overrides"] = {"name": "OtherDevice"}
    with pytest.raises(ValueError, match="overrides"):
        OptimizerConfig(**data)


def test_keeper_fixtures_validate():
    from pathlib import Path
    import yaml

    fixtures = list(
        (Path(__file__).parent / "fixtures/optimizer_configs").glob("*.yaml")
    )
    assert len(fixtures) == 6
    for path in fixtures:
        config = OptimizerConfig.model_validate(yaml.safe_load(path.read_text()))
        assert config.run.max_iterations == 30
