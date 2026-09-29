"""The kind → class registry stays in step with the schema's analyzer union."""

from __future__ import annotations

import inspect

import pytest
from geecs_schemas.analysis import ANALYZER_SPECS

from image_analysis.config.registry import (
    ANALYZER_CLASS_PATHS,
    FRAME_KINDS,
    analyzer_class,
    import_class_path,
)

#: Kinds whose modules hard-import a vendor SDK / DLL wrapper that is not
#: installed on a plain development host — resolvable in principle, skipped
#: here when the import fails for that reason.
VENDOR_KINDS = {"haso"}


def test_registry_covers_exactly_the_frame_scoped_schema_kinds():
    assert set(ANALYZER_CLASS_PATHS) == FRAME_KINDS
    assert FRAME_KINDS == {k for k, m in ANALYZER_SPECS.items() if m.scope == "frame"}
    assert set(ANALYZER_SPECS) - FRAME_KINDS == {"himg_to_stack"}


def test_unknown_kind_is_a_keyerror_naming_the_known_kinds():
    with pytest.raises(KeyError, match="known kinds"):
        analyzer_class("no_such_kind")


def test_scan_scoped_kind_has_no_image_analyzer():
    with pytest.raises(KeyError, match="scan-scoped"):
        analyzer_class("himg_to_stack")


@pytest.mark.parametrize("kind", sorted(FRAME_KINDS))
def test_every_kind_resolves_to_a_class_accepting_its_spec(kind):
    try:
        cls = analyzer_class(kind)
    except ImportError as exc:
        if kind in VENDOR_KINDS:
            pytest.skip(f"vendor SDK not installed: {exc}")
        raise
    assert inspect.isclass(cls)
    params = inspect.signature(cls.__init__).parameters
    spec_model = ANALYZER_SPECS[kind]
    has_params = set(spec_model.model_fields) - {"kind"}
    if has_params:
        assert "spec" in params, f"{cls.__name__} must accept spec= for kind {kind!r}"
    image_kind = spec_model.image_kind
    if image_kind == "camera":
        assert "camera_config" in params
    elif image_kind == "line":
        assert "line_config" in params
    else:
        assert "camera_config" not in params and "line_config" not in params


def test_import_class_path_reports_missing_module():
    with pytest.raises(ImportError, match="Cannot import module"):
        import_class_path("no.such.module.Class")
