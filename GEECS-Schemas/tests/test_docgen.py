"""Docgen tests — these are the no-drift guard for operator documentation.

A model field without a plain-language description fails here, which fails
CI, which is the mechanism keeping GUI tooltips and generated reference docs
from drifting away from the code.
"""

import pytest

from geecs_schemas import SCHEMA_REGISTRY
from geecs_schemas.docgen import (
    EXAMPLES,
    iter_nested_models,
    render_model_markdown,
    render_reference,
)


def all_models():
    seen = []
    for model in SCHEMA_REGISTRY.values():
        for nested in iter_nested_models(model):
            if nested not in seen:
                seen.append(nested)
    return seen


@pytest.mark.parametrize("model", all_models(), ids=lambda m: m.__name__)
class TestEveryModel:
    def test_every_field_has_operator_description(self, model):
        missing = [
            name
            for name, field in model.model_fields.items()
            if not (field.description or "").strip()
        ]
        assert not missing, (
            f"{model.__name__} fields {missing} have no operator-language "
            "description — add description= to the Field()."
        )

    def test_has_operator_docstring(self, model):
        assert (model.__doc__ or "").strip(), (
            f"{model.__name__} has no docstring — the first paragraph must "
            "explain the model in operator language."
        )

    def test_renders_markdown(self, model):
        markdown = render_model_markdown(model)
        assert f"### {model.__name__}" in markdown
        assert "| Field |" in markdown


class TestReference:
    def test_full_reference_renders(self):
        reference = render_reference()
        for kind, model in SCHEMA_REGISTRY.items():
            assert f"## `{kind}`" in reference
            assert f"### {model.__name__}" in reference

    def test_every_registry_kind_has_example(self):
        assert set(EXAMPLES) == set(SCHEMA_REGISTRY)

    def test_examples_validate_against_their_models(self):
        yaml = pytest.importorskip("yaml")
        for kind, model in SCHEMA_REGISTRY.items():
            document = yaml.safe_load(EXAMPLES[kind])
            model.model_validate(document)
