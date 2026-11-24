"""What a kind's form draws beyond its schema: the layout hints the theme's ``form.js`` reads.

The schema served at ``/api/schema/{kind}`` is the kind's (or, for an
analysis kind, its consumer's: the portal's, a recipe's steps and measure
bound to the analysis core's registry); this module adds the hints that say
how the editor lays it out, never what is valid. ``form.js`` reads them
generally (its docstring is their contract):

- ``x-sections`` on the root: titled folds in reading order, each with its
  keys, a help line, a ``note`` template and an ``advanced`` fold;
- ``x-ndim-from`` on the root: where a document's frame dimensionality is
  read (the input's kind, each kind's ``ndim`` the schema's own), so a
  choice offers the variants that fit it (``x-ndim`` on each, from the
  registry) and names the frames a variant is for;
- ``x-suggest`` on a word: the names a group's entry may reference;
- ``x-form-note`` on the root: why the form is plainer than it can be (the
  registry did not reach it).
"""

from __future__ import annotations

import copy
from typing import Any, get_args

from geecs_schemas.analysis import RecipeInput

Json = Any  # Any: a JSON Schema fragment

#: The recipe form in reading order: what is read, how each frame is
#: processed, what is measured, how it is drawn, the scan-level figures, how
#: the run behaves.
RECIPE_SECTIONS: list[dict[str, Json]] = [
    {
        "title": "Source",
        "keys": [
            "device",
            "output_name",
            "scalar_suffix",
            "description",
            "input",
            "inputs",
        ],  # fmt: skip
        "gist": ["device", "output_name", "scalar_suffix", "input", "inputs"],
        "help": "The device whose folder is read, how one frame is read, and any frame loaded before the run for a step or the measure to use by name (a background image, a haso reference). An input folder analysis:<folder> reads an earlier analysis's per-shot files from analysis/ScanNNN/<folder>/ instead (analysis:<folder>/<series> when it holds several).",  # fmt: skip
        # the recipe's own naming rules (AnalysisRecipe.data_folder, effective_output_name), pinned to them by test
        "note": "Reads scans/ScanNNN/{input.folder|device}/; writes {output_name|device}_<scalar>{scalar_suffix} s-file columns and analysis/ScanNNN/{output_name|device}/.",  # fmt: skip
        "advanced": {
            "title": "Naming overrides",
            "keys": ["output_name", "scalar_suffix"],
            "help": "Optional. Rename what this recipe writes (a readable label, or two recipes on one device), or tag its s-file columns (a variant of the same analysis). Unset, everything is named after the device.",  # fmt: skip
        },
    },
    {
        "title": "Steps",
        "keys": ["steps"],
        "help": "Processing in order, top to bottom; a step may repeat. Only the steps that fit this recipe's frames are offered.",
    },  # fmt: skip
    {
        "title": "Measure",
        "keys": ["measure"],
        "help": "What is measured on every processed frame; its scalars become s-file columns.",
    },  # fmt: skip
    {
        "title": "Figure",
        "keys": ["figure"],
        "help": "The per-frame draw, reused by every product image and summary panel: matplotlib keywords by call (imshow, plot, colorbar, axes, fig) and a style per overlay id. Numbers, true / false and [lists] are typed; other text is a word.",
    },  # fmt: skip
    {
        "title": "Summaries",
        "keys": ["summaries"],
        "help": "Scan-level figures, each a fixed kind with its own options; an empty list draws none.",
    },  # fmt: skip
    {
        "title": "Scan",
        "keys": ["scan", "metadata"],
        "help": "How the recipe runs over a scan, and free-form notes nothing reads.",
    },  # fmt: skip
]

#: What a frame of each dimensionality is called beside a variant that fits only one.
FRAME_NAMES = {"1": "traces", "2": "images"}


def _input_ndims() -> dict[str, int]:
    """Each recipe input kind's frame dimensionality: the input model's own ``ndim``."""
    union = get_args(RecipeInput)[0]
    return {cls.model_fields["kind"].default: cls.ndim for cls in get_args(union)}


def recipe_hints(schema: Json, *, registry: bool, source: str) -> Json:
    """A recipe's schema with its layout hints; without the registry, the note saying why it is plainer."""
    out = copy.deepcopy(schema)
    out["x-sections"] = copy.deepcopy(RECIPE_SECTIONS)
    out["x-ndim-from"] = {"path": ["input", "kind"], "values": _input_ndims(), "names": dict(FRAME_NAMES)}  # fmt: skip
    if not registry:
        out["x-form-note"] = (
            f"The analysis registry did not reach this form ({source}): a step and the "
            "measure are typed by name, their parameters as key / value rows. With the "
            "portal attached they are lists of the registered names, each with its "
            "parameters, its reference and the frames it fits."
        )
    return out


def group_hints(schema: Json, names: list[str]) -> Json:
    """A group's schema with the analyzers-tree names its entries may reference, offered as words."""
    out = copy.deepcopy(schema)
    ref = out.get("$defs", {}).get("AnalyzerRef", {}).get("properties", {}).get("ref")
    if ref is not None:
        ref["x-suggest"] = sorted(set(names))
    return out
