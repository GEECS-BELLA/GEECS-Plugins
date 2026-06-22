"""Tests for ``image_analysis.utils``.

The generic file readers moved to ``geecs_data_utils.io.images``. These tests
pin the backwards-compatibility shim: importing the readers from the old
location still works but emits a ``DeprecationWarning`` and resolves to the
same function objects as the new location.
"""

from __future__ import annotations

import warnings

import pytest

from geecs_data_utils.io import images

MOVED_READERS = [
    "read_imaq_image",
    "read_imaq_png_image",
    "read_tsv_file",
    "load_image_from_h5",
]


@pytest.mark.parametrize("name", MOVED_READERS)
def test_relocated_reader_warns_and_resolves(name):
    """Accessing a moved reader via the shim warns and returns the new object."""
    import image_analysis.utils as utils

    with pytest.warns(DeprecationWarning, match="geecs_data_utils.io.images"):
        shimmed = getattr(utils, name)

    assert shimmed is getattr(images, name)


def test_unknown_attribute_still_raises():
    """The shim does not mask genuinely missing attributes."""
    import image_analysis.utils as utils

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any spurious warning would fail here
        with pytest.raises(AttributeError):
            utils.does_not_exist
