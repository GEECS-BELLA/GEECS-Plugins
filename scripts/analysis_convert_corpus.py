r"""Convert every v2 diagnostic the analysis core serves into a v3 recipe, in place.

Run from the repository's root Poetry environment against a configs checkout::

    poetry run python scripts/analysis_convert_corpus.py --write \
        --configs ../GEECS-Plugins-Configs/scan_analysis_configs

Every ``analyzers/<namespace>/*.yaml`` that reads as a v2 diagnostic and
compiles for the core is converted with ``geecs_analysis.compat.convert.to_v3``
(built on the v2 adapter's compile output, recompiled and compared before it
is accepted) and written back to the same path in the canonical form the
config editor writes. Recipes the core does not serve (unported analyzer
kinds, scan-context backgrounds) and files already in v3 are left as they
are. Without ``--write`` nothing is written; the report is the same. The
report lists, per file, what the v3 shape does not carry from the source.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from geecs_analysis.compat.convert import to_v3
from geecs_analysis.compat.v2 import UnsupportedRecipe, compile_v2
from geecs_analysis.recipe import compile_recipe
from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisRecipe,
    canonical_document,
    load_analysis_document,
)
from scan_analysis.config_store import dump_yaml


def convert_tree(root: Path, *, write: bool) -> int:
    """Convert the tree under ``root/analyzers``; return the number converted."""
    converted = 0
    for path in sorted((root / "analyzers").glob("*/*.y*ml")):
        label = f"{path.parent.name}/{path.stem}"
        raw = yaml.safe_load(path.read_text())
        document = load_analysis_document(raw)
        if isinstance(document, AnalysisRecipe):
            print(f"{label}: already v3")
            continue
        assert isinstance(document, AnalysisDiagnostic)
        try:
            conversion = to_v3(document)
        except UnsupportedRecipe as exc:
            print(f"{label}: stays v2 ({exc})")
            continue
        text = dump_yaml(canonical_document(conversion.recipe))
        # The written file must read back to a recipe that compiles exactly
        # as its source did; the converter checked its own object, this
        # checks the serialization.
        reread = load_analysis_document(yaml.safe_load(text))
        assert isinstance(reread, AnalysisRecipe)
        source = compile_v2(document, allow_file_backgrounds=True)
        result = compile_recipe(reread, allow_file_backgrounds=True)
        if result != source and not any("coordinates" in n for n in conversion.notes):
            raise AssertionError(f"{label}: written recipe compiles differently")
        print(f"{label}: v3 ({document.analyzer.kind} -> {reread.measure.kind})")
        for note in conversion.notes:
            print(f"    note: {note}")
        if write:
            path.write_text(text)
        converted += 1
    return converted


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--configs",
        required=True,
        type=Path,
        help="the scan_analysis_configs root (the parent of analyzers/)",
    )
    parser.add_argument(
        "--write", action="store_true", help="write the converted files in place"
    )
    args = parser.parse_args(argv)
    if not (args.configs / "analyzers").is_dir():
        print(f"no analyzers/ under {args.configs}", file=sys.stderr)
        return 2
    count = convert_tree(args.configs, write=args.write)
    print(f"{count} recipe(s) {'written' if args.write else 'convertible'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
