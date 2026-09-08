"""``Data1DLoading`` (GEECS-Schemas) mirrors ``Data1DConfig`` (GEECS-Data-Utils) field for field.

The schema package cannot import GEECS-Data-Utils, so the two models are
kept in step by this test — the one place both are importable.
"""

from geecs_data_utils.io.array1d import Data1DConfig, Data1DType as UtilsType
from geecs_schemas.analysis import Data1DLoading, Data1DType

from image_analysis.data_1d_utils import to_data1d_config


def test_field_names_and_defaults_match():
    a, b = Data1DLoading.model_fields, Data1DConfig.model_fields
    assert set(a) == set(b)
    for name in a:
        assert a[name].is_required() == b[name].is_required(), name
        if not a[name].is_required():
            assert a[name].get_default(call_default_factory=True) == b[
                name
            ].get_default(call_default_factory=True), name


def test_data_type_enums_match():
    assert {m.value for m in Data1DType} == {m.value for m in UtilsType}


def test_reader_config_instance_is_accepted_and_converted_back():
    reader_cfg = Data1DConfig(data_type="tsv", x_column=2, auxiliary_columns={"w": 3})
    loading = Data1DLoading.model_validate(reader_cfg)
    assert loading.x_column == 2 and loading.auxiliary_columns == {"w": 3}
    back = to_data1d_config(loading)
    assert isinstance(back, Data1DConfig)
    assert back.model_dump(mode="json") == reader_cfg.model_dump(mode="json")


def test_negative_auxiliary_column_is_refused_at_the_schema_layer():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Data1DLoading(data_type="tsv", auxiliary_columns={"w": -1})
